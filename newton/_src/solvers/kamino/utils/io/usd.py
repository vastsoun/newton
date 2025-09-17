###########################################################################
# KAMINO: Utilities: Input/Output: OpenUSD
###########################################################################

import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
import warp as wp

import newton._src.solvers.kamino.utils.logger as msg
from newton._src.core.types import nparray
from newton._src.solvers.kamino.core.bodies import RigidBodyDescriptor
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.geometry import CollisionGeometryDescriptor, GeometryDescriptor
from newton._src.solvers.kamino.core.gravity import GravityDescriptor
from newton._src.solvers.kamino.core.joints import JointActuationType, JointDescriptor, JointDoFType
from newton._src.solvers.kamino.core.materials import (
    DEFAULT_DENSITY,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    MaterialDescriptor,
    MaterialPairProperties,
)
from newton._src.solvers.kamino.core.math import I_3, screw
from newton._src.solvers.kamino.core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    PlaneShape,
    # ConvexShape,
    # MeshShape,
    # SDFShape
    SphereShape,
)
from newton._src.solvers.kamino.core.types import Axis, AxisType, Transform, quatf, transformf, vec3f

###
# Helper Functions
###

__axis_rotations = {}


def quat_between_axes(*axes: AxisType) -> quatf:
    """
    Returns a quaternion that represents the rotations between the given sequence of axes.

    Args:
        axes (AxisType): The axes between to rotate.

    Returns:
        wp.quat: The rotation quaternion.
    """
    q = wp.quat_identity()
    for i in range(len(axes) - 1):
        src = Axis.from_any(axes[i])
        dst = Axis.from_any(axes[i + 1])
        if (src.value, dst.value) in __axis_rotations:
            dq = __axis_rotations[(src.value, dst.value)]
        else:
            dq = wp.quat_between_vectors(src.to_vec3(), dst.to_vec3())
            __axis_rotations[(src.value, dst.value)] = dq
        q *= dq
    return q


###
# Importer
###


class USDImporter:
    """
    A class to parse OpenUSD files and extract relevant data.
    """

    # Class-level variable to hold the imported modules
    Sdf = None
    Usd = None
    UsdGeom = None
    UsdPhysics = None

    @classmethod
    def _load_pxr_openusd(cls):
        """
        Attempts to import the necessary USD modules.
        Raises ImportError if the modules cannot be imported.
        """
        if cls.Sdf is None:
            try:
                from pxr import Sdf, Usd, UsdGeom, UsdPhysics

                cls.Sdf = Sdf
                cls.Usd = Usd
                cls.UsdGeom = UsdGeom
                cls.UsdPhysics = UsdPhysics
            except ImportError as e:
                raise ImportError("Failed to import pxr. Please install USD (e.g. via `pip install usd-core`).") from e

    def __init__(self):
        # Load the necessary USD modules
        self._load_pxr_openusd()
        self._loaded_pxr: bool = True
        self._invert_rotations: bool = False

        # Define the axis mapping from USD
        self.usd_axis_to_axis = {
            self.UsdPhysics.Axis.X: Axis.X,
            self.UsdPhysics.Axis.Y: Axis.Y,
            self.UsdPhysics.Axis.Z: Axis.Z,
        }

        # Define the axis mapping from USD
        self.usd_dofs_to_axis = {
            self.UsdPhysics.JointDOF.TransX: Axis.X,
            self.UsdPhysics.JointDOF.TransY: Axis.Y,
            self.UsdPhysics.JointDOF.TransZ: Axis.Z,
            self.UsdPhysics.JointDOF.RotX: Axis.X,
            self.UsdPhysics.JointDOF.RotY: Axis.Y,
            self.UsdPhysics.JointDOF.RotZ: Axis.Z,
        }

        # Define the joint DoF axes for translations and rotations
        self._usd_trans_axes = (
            self.UsdPhysics.JointDOF.TransX,
            self.UsdPhysics.JointDOF.TransY,
            self.UsdPhysics.JointDOF.TransZ,
        )
        self._usd_rot_axes = (
            self.UsdPhysics.JointDOF.RotX,
            self.UsdPhysics.JointDOF.RotY,
            self.UsdPhysics.JointDOF.RotZ,
        )

        # Define the supported USD joint types
        self.supported_usd_joint_types = (
            self.UsdPhysics.ObjectType.FixedJoint,
            self.UsdPhysics.ObjectType.RevoluteJoint,
            self.UsdPhysics.ObjectType.PrismaticJoint,
            self.UsdPhysics.ObjectType.SphericalJoint,
            self.UsdPhysics.ObjectType.D6Joint,
        )
        self.supported_usd_joint_type_names = (
            "PhysicsFixedJoint",
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
            "PhysicsSphericalJoint",
            "PhysicsJoint",
        )

        # TODO: Add support for non-physics geoms
        # Define the supported USD geom types
        self.supported_usd_geom_types = (
            self.UsdPhysics.ObjectType.CapsuleShape,
            self.UsdPhysics.ObjectType.Capsule1Shape,
            self.UsdPhysics.ObjectType.ConeShape,
            self.UsdPhysics.ObjectType.CubeShape,
            self.UsdPhysics.ObjectType.CylinderShape,
            self.UsdPhysics.ObjectType.Cylinder1Shape,
            self.UsdPhysics.ObjectType.PlaneShape,
            self.UsdPhysics.ObjectType.SphereShape,
            self.UsdPhysics.ObjectType.MeshShape,
        )
        self.supported_usd_geom_type_names = (
            "Capsule",
            "Capsule1",
            "Cone",
            "Cube",
            "Cylinder",
            "Cylinder1",
            "Plane",
            "Sphere",
            "Mesh",
        )

    ###
    # Back-end Functions
    ###

    @staticmethod
    def _get_prim_name(prim) -> str:
        """Retrieves the name of the prim from its path."""
        return str(prim.GetPath())[len(str(prim.GetParent().GetPath())) :].lstrip("/")

    @staticmethod
    def _get_prim_uid(prim) -> str:
        """Queries the custom data for a unique identifier (UID)."""
        uid = None
        cdata = prim.GetCustomData()
        if cdata is not None:
            uid = cdata.get("uuid", None)
        return uid if uid is not None else str(uuid.uuid4())

    @staticmethod
    def _get_material_default_override(prim) -> bool:
        """Queries the custom data to detect if the prim should override the default material."""
        override_default = False
        cdata = prim.GetCustomData()
        if cdata is not None:
            override_default = cdata.get("overrideDefault", False)
        return override_default

    def _get_attribute(self, prim, name) -> Any:
        return prim.GetAttribute(name)

    def _has_attribute(self, prim, name) -> bool:
        attr = self._get_attribute(prim, name)
        return attr.IsValid() and attr.HasAuthoredValue()

    def _parse_float(self, prim, name, default=None) -> float | None:
        attr = self._get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val):
            return val
        return default

    def _parse_float_with_fallback(self, prims: Iterable[Any], name: str, default: float = 0.0) -> float:
        ret = default
        for prim in prims:
            if not prim:
                continue
            attr = self._get_attribute(prim, name)
            if not attr or not attr.HasAuthoredValue():
                continue
            val = attr.Get()
            if np.isfinite(val):
                ret = val
                break
        return ret

    @staticmethod
    def _from_gfquat(gfquat) -> wp.quat:
        return wp.normalize(wp.quat(*gfquat.imaginary, gfquat.real))

    def _parse_quat(self, prim, name, default=None) -> nparray | None:
        attr = self._get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if self._invert_rotations:
            quat = wp.quat(*val.imaginary, -val.real)
        else:
            quat = wp.quat(*val.imaginary, val.real)
        qn = wp.length(quat)
        if np.isfinite(qn) and qn > 0.0:
            return quat
        return default

    def _parse_vec(self, prim, name, default=None) -> nparray | None:
        attr = self._get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        val = attr.Get()
        if np.isfinite(val).all():
            return np.array(val, dtype=np.float32)
        return default

    def _parse_generic(self, prim, name, default=None) -> Any | None:
        attr = self._get_attribute(prim, name)
        if not attr or not attr.HasAuthoredValue():
            return default
        return attr.Get()

    def _parse_xform(self, prim) -> wp.transform:
        xform = self.UsdGeom.Xform(prim)
        mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
        if self._invert_rotations:
            rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
        else:
            rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].flatten()))
        pos = mat[3, :3]
        return wp.transform(pos, rot)

    def _get_geom_max_contacts(self, prim) -> int:
        """Queries the custom data for the max contacts hint."""
        max_contacts = None
        cdata = prim.GetCustomData()
        if cdata is not None:
            max_contacts = cdata.get("maxContacts", None)
        return int(max_contacts) if max_contacts is not None else 0

    @staticmethod
    def _material_pair_properties_from(first: MaterialDescriptor, second: MaterialDescriptor) -> MaterialPairProperties:
        pair_properties = MaterialPairProperties()
        pair_properties.restitution = 0.5 * (first.restitution + second.restitution)
        pair_properties.static_friction = 0.5 * (first.static_friction + second.static_friction)
        pair_properties.dynamic_friction = 0.5 * (first.dynamic_friction + second.dynamic_friction)
        return pair_properties

    def _parse_material(
        self,
        material_prim,
        distance_unit: float = 1.0,
        mass_unit: float = 1.0,
    ) -> MaterialDescriptor | None:
        """
        Parses a material prim and returns a MaterialDescriptor.

        Args:
            material_prim: The USD prim representing the material.
            material_spec: The UsdPhysicsRigidBodyMaterialDesc entry.
            distance_unit: The global unit of distance of the USD stage.
            mass_unit: The global unit of mass of the USD stage.
        """

        # Retrieve the namespace path of the prim
        path = str(material_prim.GetPath())
        msg.info(f"path: {path}")

        # Define and check for the required APIs
        req_api = ["PhysicsMaterialAPI"]
        for api in req_api:
            if api not in material_prim.GetAppliedSchemas():
                raise ValueError(
                    f"Required API '{api}' not found on prim '{path}'. "
                    "Please ensure the prim has the necessary schemas applied."
                )

        ###
        # Prim Identifiers
        ###

        # Retrieve the name and UID of the rigid body from the prim
        name = self._get_prim_name(material_prim)
        uid = self._get_prim_uid(material_prim)
        msg.info(f"name: {name}")
        msg.info(f"uid: {uid}")

        ###
        # Material Properties
        ###

        # Retrieve the USD material properties
        density_scale = mass_unit / distance_unit**3
        density = (density_scale) * self._parse_float(material_prim, "physics:density", default=DEFAULT_DENSITY)
        restitution = self._parse_float(material_prim, "physics:restitution", default=DEFAULT_RESTITUTION)
        static_friction = self._parse_float(material_prim, "physics:staticFriction", default=DEFAULT_FRICTION)
        dynamic_friction = self._parse_float(material_prim, "physics:dynamicFriction", default=DEFAULT_FRICTION)
        msg.info(f"density: {density}")
        msg.info(f"restitution: {restitution}")
        msg.info(f"static_friction: {static_friction}")
        msg.info(f"dynamic_friction: {dynamic_friction}")

        ###
        # MaterialDescriptor
        ###

        return MaterialDescriptor(
            name=name,
            uid=uid,
            density=density,
            restitution=restitution,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )

    def _parse_rigid_body(
        self,
        rigid_body_prim,
        rigid_body_spec,
        distance_unit: float = 1.0,
        rotation_unit: float = 1.0,
        mass_unit: float = 1.0,
        offset_xform: wp.transform | None = None,
        only_load_enabled_rigid_bodies: bool = True,
    ) -> RigidBodyDescriptor | None:
        # Skip this body if it is not enable and we are only loading enabled rigid bodies
        if not rigid_body_spec.rigidBodyEnabled and only_load_enabled_rigid_bodies:
            return None

        # Retrieve the namespace path of the prim
        path = str(rigid_body_prim.GetPath())

        # Define and check for the required APIs
        req_api = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        for api in req_api:
            if api not in rigid_body_prim.GetAppliedSchemas():
                raise ValueError(
                    f"Required API '{api}' not found on prim '{path}'. "
                    "Please ensure the prim has the necessary schemas applied."
                )

        ###
        # Prim Identifiers
        ###

        # Retrieve the name and UID of the rigid body from the prim
        name = self._get_prim_name(rigid_body_prim)
        uid = self._get_prim_uid(rigid_body_prim)
        msg.info(f"name: {name}")
        msg.info(f"uid: {uid}")

        ###
        # PhysicsRigidBodyAPI
        ###

        # Retrieve the rigid body origin (i.e. the pose of the body frame)
        body_xform = wp.transform(distance_unit * rigid_body_spec.position, self._from_gfquat(rigid_body_spec.rotation))

        # Apply an offset transformation to the origin if provided
        if offset_xform is not None:
            body_xform = wp.mul(distance_unit * offset_xform, body_xform)

        # Retrieve the linear and angular velocities
        # NOTE: They are transformed to world coordiates since the RigidBodyAPI specifies them in local body coordinates
        v_i = wp.transform_vector(body_xform, distance_unit * vec3f(rigid_body_spec.linearVelocity))
        omega_i = wp.transform_vector(body_xform, rotation_unit * vec3f(rigid_body_spec.angularVelocity))
        msg.info(f"body_xform: {body_xform}")
        msg.info(f"omega_i: {omega_i}")
        msg.info(f"v_i: {v_i}")

        ###
        # PhysicsMassAPI
        ###

        # Extract the mass, center of mass, diagonal inertia, and principal axes from the prim
        m_i = mass_unit * self._parse_float(rigid_body_prim, "physics:mass")
        i_r_com_i = distance_unit * self._parse_vec(rigid_body_prim, "physics:centerOfMass")
        i_I_i_diag = mass_unit * self._parse_vec(rigid_body_prim, "physics:diagonalInertia")
        i_q_i_pa = self._parse_quat(rigid_body_prim, "physics:principalAxes")
        msg.info(f"m_i: {m_i}")
        msg.info(f"i_r_com_i: {i_r_com_i}")
        msg.info(f"i_I_i_diag: {i_I_i_diag}")
        msg.info(f"i_q_i_pa: {i_q_i_pa}")

        # Check if the required properties are defined
        if m_i is None:
            raise ValueError(f"Rigid body '{path}' has no mass defined. Please set the mass using 'physics:mass'.")
        if i_r_com_i is None:
            raise ValueError(
                f"Rigid body '{path}' has no center of mass defined. Please set the center of mass using 'physics:centerOfMass'."
            )
        if i_I_i_diag is None:
            raise ValueError(
                f"Rigid body '{path}' has no diagonal inertia defined. Please set the diagonal inertia using 'physics:diagonalInertia'."
            )
        if i_q_i_pa is None:
            raise ValueError(
                f"Rigid body '{path}' has no principal axes defined. Please set the principal axes using 'physics:principalAxes'."
            )

        # Check each property to ensure they are valid
        # TODO: What should we check?
        # TODO: Should we handle massless bodies?

        # Compute the moment of inertia matrix (in body-local coordiantes) from the diagonal inertia and principal axes
        i_I_i_diag = wp.diag(vec3f(i_I_i_diag))
        i_q_i_pa = wp.normalize(quatf(i_q_i_pa))
        R_i_pa = wp.quat_to_matrix(i_q_i_pa)
        i_I_i = R_i_pa @ i_I_i_diag @ wp.transpose(R_i_pa)
        msg.info(f"i_I_i_diag:\n{i_I_i_diag}")
        msg.info(f"i_q_i_pa: {i_q_i_pa}")
        msg.info(f"R_i_pa:\n{R_i_pa}")
        msg.info(f"i_I_i:\n{i_I_i}")

        # Compute the center of mass in world coordinates
        r_com_i = wp.transform_point(body_xform, vec3f(i_r_com_i))
        msg.info(f"r_com_i: {r_com_i}")

        # Construc the initial pose and twist of the body in world coordinates
        q_i_0 = transformf(r_com_i, body_xform.q)
        u_i_0 = screw(v_i, omega_i)
        msg.info(f"q_i_0: {q_i_0}")
        msg.info(f"u_i_0: {u_i_0}")

        ###
        # RigidBodyDescriptor
        ###

        # Construct and return the RigidBodyDescriptor
        # with the data imported from the USD prim
        body_desc = RigidBodyDescriptor()
        body_desc.name = name
        body_desc.uid = uid
        body_desc.m_i = m_i
        body_desc.i_I_i = i_I_i
        body_desc.q_i_0 = q_i_0
        body_desc.u_i_0 = u_i_0
        return body_desc

    def _has_joints(self, ret_dict: dict) -> bool:
        """
        Check if the ret_dict contains any joints.
        """
        for joint_type in self.supported_usd_joint_types:
            if joint_type in ret_dict:
                return True
        return False

    def _get_joint_dof_hint(self, prim) -> JointDoFType | None:
        """Queries the custom data for a DoF type hints."""
        dofs = None
        cdata = prim.GetCustomData()
        if cdata is not None:
            dofs = cdata.get("dofs", None)
        dof_type = None
        if dofs == "cylindrical":
            dof_type = JointDoFType.CYLINDRICAL
        elif dofs == "universal":
            dof_type = JointDoFType.UNIVERSAL
        elif dofs == "cartesian":
            dof_type = JointDoFType.CARTESIAN
        return dof_type

    def _parse_joint_revolute(self, joint_spec, rotation_unit: float = 1.0):
        dof_type = JointDoFType.REVOLUTE
        X_j = self.usd_axis_to_axis[joint_spec.axis].to_mat33()
        q_j_min = None
        q_j_max = None
        tau_j_max = None
        if joint_spec.limit.enabled:
            q_j_min = rotation_unit * joint_spec.limit.lower
            q_j_max = rotation_unit * joint_spec.limit.upper
        if joint_spec.drive.enabled:
            if not joint_spec.drive.acceleration:
                act_type = JointActuationType.FORCE
                tau_j_max = joint_spec.drive.forceLimit
            else:
                # TODO: Should we handle acceleration drives?
                raise ValueError("Revolute acceleration drive actuators are not yet supported.")
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max

    def _parse_joint_prismatic(self, joint_spec, distance_unit: float = 1.0):
        dof_type = JointDoFType.PRISMATIC
        X_j = self.usd_axis_to_axis[joint_spec.axis].to_mat33()
        q_j_min = None
        q_j_max = None
        tau_j_max = None
        if joint_spec.limit.enabled:
            q_j_min = distance_unit * joint_spec.limit.lower
            q_j_max = distance_unit * joint_spec.limit.upper
        if joint_spec.drive.enabled:
            if not joint_spec.drive.acceleration:
                act_type = JointActuationType.FORCE
                tau_j_max = joint_spec.drive.forceLimit
            else:
                # TODO: Should we handle acceleration drives?
                raise ValueError("Prismatic acceleration drive actuators are not yet supported.")
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max

    def _parse_joint_revolute_from_d6(self, name, joint_prim, joint_spec, joint_dof, rotation_unit: float = 1.0):
        dof_type = JointDoFType.REVOLUTE
        X_j = self.usd_dofs_to_axis[joint_dof].to_mat33()
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == joint_dof:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != 1:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives} drives, "
                    "but revolute joints require exactly one drive."
                )
            act_type = JointActuationType.FORCE
            tau_j_max = joint_spec.jointDrives[0].second.forceLimit
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max

    def _parse_joint_prismatic_from_d6(self, name, joint_prim, joint_spec, joint_dof, distance_unit: float = 1.0):
        dof_type = JointDoFType.PRISMATIC
        X_j = self.usd_dofs_to_axis[joint_dof].to_mat33()
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == joint_dof:
                q_j_min.append(distance_unit * limit.second.lower)
                q_j_max.append(distance_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != 1:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives} drives, "
                    "but prismatic joints require exactly one drive."
                )
            act_type = JointActuationType.FORCE
            tau_j_max = joint_spec.jointDrives[0].second.forceLimit
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max

    def _parse_joint_cylindrical_from_d6(
        self, name, joint_prim, joint_spec, distance_unit: float = 1.0, rotation_unit: float = 1.0
    ):
        dof_type = JointDoFType.CYLINDRICAL
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == self.UsdPhysics.JointDOF.TransX:
                q_j_min.append(distance_unit * limit.second.lower)
                q_j_max.append(distance_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.RotX:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != JointDoFType.CYLINDRICAL.num_dofs:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives}"
                    f"drives, but cylindrical joints require {JointDoFType.CYLINDRICAL.num_dofs} drives. "
                )
            act_type = JointActuationType.FORCE
            tau_j_max = [drive.second.forceLimit for drive in joint_spec.jointDrives]
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, q_j_min, q_j_max, tau_j_max

    def _parse_joint_universal_from_d6(self, name, joint_prim, joint_spec, rotation_unit: float = 1.0):
        dof_type = JointDoFType.UNIVERSAL
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == self.UsdPhysics.JointDOF.RotX:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.RotY:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != JointDoFType.UNIVERSAL.num_dofs:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives}"
                    f"drives, but universal joints require {JointDoFType.UNIVERSAL.num_dofs} drives. "
                )
            act_type = JointActuationType.FORCE
            tau_j_max = [drive.second.forceLimit for drive in joint_spec.jointDrives]
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, q_j_min, q_j_max, tau_j_max

    def _parse_joint_cartesian_from_d6(
        self,
        name,
        joint_prim,
        joint_spec,
        distance_unit: float = 1.0,
    ):
        dof_type = JointDoFType.CARTESIAN
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == self.UsdPhysics.JointDOF.TransX:
                q_j_min.append(distance_unit * limit.second.lower)
                q_j_max.append(distance_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.TransY:
                q_j_min.append(distance_unit * limit.second.lower)
                q_j_max.append(distance_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.TransZ:
                q_j_min.append(distance_unit * limit.second.lower)
                q_j_max.append(distance_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != JointDoFType.CARTESIAN.num_dofs:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives}"
                    f"drives, but cartesian joints require {JointDoFType.CARTESIAN.num_dofs} drives. "
                )
            act_type = JointActuationType.FORCE
            tau_j_max = [drive.second.forceLimit for drive in joint_spec.jointDrives]
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, q_j_min, q_j_max, tau_j_max

    def _parse_joint_spherical_from_d6(self, name, joint_prim, joint_spec, rotation_unit: float = 1.0):
        dof_type = JointDoFType.SPHERICAL
        q_j_min = []
        q_j_max = []
        tau_j_max = None
        for limit in joint_spec.jointLimits:
            dof = limit.first
            if dof == self.UsdPhysics.JointDOF.RotX:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.RotY:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
            elif dof == self.UsdPhysics.JointDOF.RotZ:
                q_j_min.append(rotation_unit * limit.second.lower)
                q_j_max.append(rotation_unit * limit.second.upper)
        num_drives = len(joint_spec.jointDrives)
        if num_drives > 0:
            if num_drives != JointDoFType.SPHERICAL.num_dofs:
                raise ValueError(
                    f"Joint '{name}' ({joint_prim.GetPath()}) has {num_drives}"
                    f"drives, but spherical joints require {JointDoFType.SPHERICAL.num_dofs} drives. "
                )
            act_type = JointActuationType.FORCE
            tau_j_max = [drive.second.forceLimit for drive in joint_spec.jointDrives]
        else:
            act_type = JointActuationType.PASSIVE
        return dof_type, act_type, q_j_min, q_j_max, tau_j_max

    def _parse_joint(
        self,
        joint_prim,
        joint_spec,
        joint_type,
        body_index_map: dict[str, int],
        distance_unit: float = 1.0,
        rotation_unit: float = 1.0,
        only_load_enabled_joints: bool = True,
    ) -> JointDescriptor | None:
        # Skip this body if it is not enable and we are only loading enabled rigid bodies
        if not joint_spec.jointEnabled and only_load_enabled_joints:
            return None

        ###
        # Prim Identifiers
        ###

        # Retrieve the name and UID of the joint from the prim
        name = self._get_prim_name(joint_prim)
        uid = self._get_prim_uid(joint_prim)
        msg.info(f"name: {name}")
        msg.info(f"uid: {uid}")

        ###
        # PhysicsJoint Common Properties
        ###

        # Check if body0 and body1 are specified
        if (not joint_spec.body0) and (not joint_spec.body1):
            raise ValueError(
                f"Joint '{name}' ({joint_prim.GetPath()}) does not specify bodies. "
                "Specify the joint bodies using 'physics:body0' and 'physics:body1'."
            )

        # Extract the relative poses of the joint
        B_r_Bj = distance_unit * vec3f(joint_spec.localPose0Position)
        F_r_Fj = distance_unit * vec3f(joint_spec.localPose1Position)
        B_q_Bj = self._from_gfquat(joint_spec.localPose0Orientation)
        F_q_Fj = self._from_gfquat(joint_spec.localPose1Orientation)
        msg.info(f"B_r_Bj: {B_r_Bj}")
        msg.info(f"F_r_Fj: {F_r_Fj}")
        msg.info(f"B_q_Bj: {B_q_Bj}")
        msg.info(f"F_q_Fj: {F_q_Fj}")

        # Check if body0 is specified
        if (not joint_spec.body0) and joint_spec.body1:
            # body0 is unspecified, and (0,1) are mapped to (B,F)
            bid_F = body_index_map[str(joint_spec.body1)]
            bid_B = -1
        elif joint_spec.body0 and (not joint_spec.body1):
            # body1 is unspecified, and (0,1) are mapped to (F,B)
            bid_F = body_index_map[str(joint_spec.body0)]
            bid_B = -1
            B_r_Bj, F_r_Fj = F_r_Fj, B_r_Bj
            B_q_Bj, F_q_Fj = F_q_Fj, B_q_Bj
        else:
            # Both bodies are specified, and (0,1) are mapped to (B,F)
            bid_B = body_index_map[str(joint_spec.body0)]
            bid_F = body_index_map[str(joint_spec.body1)]
        msg.info(f"bid_B: {bid_B}")
        msg.info(f"bid_F: {bid_F}")

        ###
        # PhysicsJoint Specific Properties
        ###

        X_j = I_3
        dof_type = None
        act_type = None
        q_j_min = None
        q_j_max = None
        tau_j_max = None

        if joint_type == self.UsdPhysics.ObjectType.FixedJoint:
            dof_type = JointDoFType.FIXED
            act_type = JointActuationType.PASSIVE

        elif joint_type == self.UsdPhysics.ObjectType.RevoluteJoint:
            dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max = self._parse_joint_revolute(
                joint_spec, rotation_unit=rotation_unit
            )

        elif joint_type == self.UsdPhysics.ObjectType.PrismaticJoint:
            dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max = self._parse_joint_prismatic(
                joint_spec, distance_unit=distance_unit
            )

        elif joint_type == self.UsdPhysics.ObjectType.SphericalJoint:
            dof_type = JointDoFType.SPHERICAL
            act_type = JointActuationType.PASSIVE
            X_j = self.usd_axis_to_axis[joint_spec.axis].to_mat33()

        elif joint_type == self.UsdPhysics.ObjectType.DistanceJoint:
            raise NotImplementedError("Distance joints are not yet supported.")

        elif joint_type == self.UsdPhysics.ObjectType.D6Joint:
            # First check if the joint contains a DoF type hint in the custom data
            # NOTE: The hint allows us to skip the extensive D6 joint parsing
            custom_dof_type = self._get_joint_dof_hint(joint_prim)
            if custom_dof_type:
                if custom_dof_type == JointDoFType.CYLINDRICAL:
                    dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_cylindrical_from_d6(
                        name, joint_prim, joint_spec, distance_unit, rotation_unit
                    )

                elif custom_dof_type == JointDoFType.UNIVERSAL:
                    dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_universal_from_d6(
                        name, joint_prim, joint_spec, rotation_unit
                    )

                elif custom_dof_type == JointDoFType.CARTESIAN:
                    dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_cartesian_from_d6(
                        name, joint_prim, joint_spec, distance_unit
                    )

                else:
                    raise ValueError(
                        f"Unsupported custom DoF type hint '{custom_dof_type}' for joint '{joint_prim.GetPath()}'. "
                        "Supported hints are: {'cylindrical', 'universal', 'cartesian'}."
                    )

            # If no custom DoF type hint is provided, we parse the D6 joint limits and drives
            else:
                # Parse the joint limits to determine the DoF type
                dofs = []
                cts = []
                for limit in joint_spec.jointLimits:
                    upper = limit.second.upper
                    lower = limit.second.lower
                    axis_is_free = lower < upper
                    axis = limit.first
                    if axis_is_free:
                        dofs.append(axis)
                    else:
                        cts.append(axis)

                # Attempt to detect the type of the joint based on the limits
                if len(dofs) == 0:
                    dof_type = JointDoFType.FIXED
                    act_type = JointActuationType.PASSIVE
                elif len(dofs) == 1:
                    if dofs[0] in self._usd_rot_axes:
                        dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max = self._parse_joint_revolute_from_d6(
                            name, joint_prim, joint_spec, dofs[0], rotation_unit
                        )
                    if dofs[0] in self._usd_trans_axes:
                        dof_type, act_type, X_j, q_j_min, q_j_max, tau_j_max = self._parse_joint_prismatic_from_d6(
                            name, joint_prim, joint_spec, dofs[0], distance_unit
                        )
                elif len(dofs) == 2:
                    if all(dof in self._usd_rot_axes for dof in dofs):
                        dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_universal_from_d6(
                            name, joint_prim, joint_spec, rotation_unit
                        )
                    if dofs[0] in self._usd_trans_axes and dofs[1] in self._usd_rot_axes:
                        dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_cylindrical_from_d6(
                            name, joint_prim, joint_spec, distance_unit, rotation_unit
                        )
                elif len(dofs) == 3:
                    if all(dof in self._usd_rot_axes for dof in dofs):
                        dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_spherical_from_d6(
                            name, joint_prim, joint_spec, rotation_unit
                        )
                    elif all(dof in self._usd_trans_axes for dof in dofs):
                        dof_type, act_type, q_j_min, q_j_max, tau_j_max = self._parse_joint_cartesian_from_d6(
                            name, joint_prim, joint_spec, distance_unit
                        )
                else:
                    raise ValueError(
                        f"Joint '{name}' ({joint_prim.GetPath()}) has {len(dofs)} free axes, "
                        "but D6 joints are only supported up to 3 DoFs. "
                    )

        elif joint_type == self.UsdPhysics.ObjectType.CustomJoint:
            raise NotImplementedError("Custom joints are not yet supported.")

        else:
            raise ValueError(
                f"Unsupported joint type: {joint_type}. Supported types are: {self.supported_usd_joint_types}."
            )
        msg.info(f"dof_type: {dof_type}")
        msg.info(f"act_type: {act_type}")
        msg.info(f"X_j:\n{X_j}")
        msg.info(f"q_j_min: {q_j_min}")
        msg.info(f"q_j_max: {q_j_max}")
        msg.info(f"tau_j_max: {tau_j_max}")

        ###
        # JointDescriptor
        ###

        # Construct and return the RigidBodyDescriptor
        # with the data imported from the USD prim
        joint_desc = JointDescriptor()
        joint_desc.name = name
        joint_desc.uid = uid
        joint_desc.act_type = act_type
        joint_desc.dof_type = dof_type
        joint_desc.bid_B = bid_B
        joint_desc.bid_F = bid_F
        joint_desc.B_r_Bj = B_r_Bj
        joint_desc.F_r_Fj = F_r_Fj
        joint_desc.X_j = wp.quat_to_matrix(B_q_Bj) @ X_j
        joint_desc.q_j_min = q_j_min
        joint_desc.q_j_max = q_j_max
        joint_desc.tau_j_max = tau_j_max
        return joint_desc

    def _parse_geom(
        self,
        geom_prim,
        geom_type,
        geom_spec,
        body_index_map: dict[str, int],
        cgroup_index_map: dict[str, int],
        material_index_map: dict[str, int],
        distance_unit: float = 1.0,
        rotation_unit: float = 1.0,
    ) -> CollisionGeometryDescriptor | GeometryDescriptor | None:
        """
        Parses a geometry prim and returns a GeometryDescriptor.
        """

        ###
        # Prim Identifiers
        ###

        # Retrieve the name and UID of the geometry from the prim
        name = self._get_prim_name(geom_prim)
        uid = self._get_prim_uid(geom_prim)
        msg.info(f"[Geom]: name: {name}")
        msg.info(f"[{name}]: uid: {uid}")

        ###
        # PhysicsGeom Common Properties
        ###

        # Retrieve the body index of the rigid body associated with the geom
        # NOTE: If a rigid body is not associated with the geom, the body index (bid) is
        # set to `-1` indicating that the geom belongs to the world, i.e. it is a static
        bid = body_index_map.get(str(geom_spec.rigidBody), -1)
        msg.info(f"[{name}]: bid: {bid}")

        # Extract the relative poses of the geom w.r.t the rigid body frame
        i_r_ig = distance_unit * vec3f(geom_spec.localPos)
        i_q_ig = self._from_gfquat(geom_spec.localRot)
        i_T_ig = transformf(i_r_ig, i_q_ig)
        msg.info(f"[{name}]: i_r_ig: {i_r_ig}")
        msg.info(f"[{name}]: i_q_ig: {i_q_ig}")

        ###
        # Layer Properties
        ###

        # TODO: Define a mechanism to handle multiple layers,
        # each potentially holding multiple geometries.
        lid = 0

        ###
        # PhysicsGeom Shape Properties
        ###

        # Retrive the geom scale
        # TODO: materials = geom_spec.materials
        scale = np.array(geom_spec.localScale)
        msg.info(f"[{name}]: scale: {scale}")

        # Construct the shape descriptor based on the geometry type
        shape = None
        if geom_type == self.UsdPhysics.ObjectType.CapsuleShape:
            # TODO: axis = geom_spec.axis, how can we use this?
            shape = CapsuleShape(radius=geom_spec.radius, height=2.0 * geom_spec.halfHeight)

        elif geom_type == self.UsdPhysics.ObjectType.Capsule1Shape:
            raise NotImplementedError("Capsule1Shape is not yet supported.")

        elif geom_type == self.UsdPhysics.ObjectType.ConeShape:
            # TODO: axis = geom_spec.axis, how can we use this?
            shape = ConeShape(radius=geom_spec.radius, height=2.0 * geom_spec.halfHeight)

        elif geom_type == self.UsdPhysics.ObjectType.CubeShape:
            d, w, h = 2.0 * distance_unit * vec3f(geom_spec.halfExtents)
            shape = BoxShape(depth=d, width=w, height=h)

        elif geom_type == self.UsdPhysics.ObjectType.CylinderShape:
            # TODO: axis = geom_spec.axis, how can we use this?
            shape = CylinderShape(radius=geom_spec.radius, height=2.0 * geom_spec.halfHeight)

        elif geom_type == self.UsdPhysics.ObjectType.Cylinder1Shape:
            raise NotImplementedError("Cylinder1Shape is not yet supported.")

        elif geom_type == self.UsdPhysics.ObjectType.PlaneShape:
            # TODO: get distance from geom position
            shape = PlaneShape(normal=self.usd_axis_to_axis[geom_spec.axis].to_vec3f(), distance=0.0)

        elif geom_type == self.UsdPhysics.ObjectType.SphereShape:
            if np.all(scale[0:] == scale[0]):
                shape = SphereShape(radius=distance_unit * geom_spec.radius)
            else:
                a, b, c = distance_unit * scale
                shape = EllipsoidShape(a=a, b=b, c=c)

        elif geom_type == self.UsdPhysics.ObjectType.MeshShape:
            msg.warning(f"Mesh shapes are not yet supported. Geom '{name}' ({geom_prim.GetPath()}) will be ignored.")
            return None  # TODO: Implement mesh shapes

        else:
            raise ValueError(
                f"Unsupported geometry type: {geom_type}. Supported types are: {self.supported_usd_geom_types}."
            )
        msg.info(f"[{name}]: shape: {shape}")

        ###
        # Base GeometryDescriptor
        ###

        descriptor = GeometryDescriptor()
        descriptor.name = name
        descriptor.uid = uid
        descriptor.lid = lid
        descriptor.bid = bid
        descriptor.offset = i_T_ig
        descriptor.shape = shape

        ###
        # Collision Properties
        ###

        # Promote the GeometryDescriptor to a CollisionGeometryDescriptor if it's collidable
        if geom_spec.collisionEnabled:
            descriptor = CollisionGeometryDescriptor(base=descriptor)

            # Query the geom prim for the maximum number of contacts hint
            descriptor.max_contacts = self._get_geom_max_contacts(geom_prim)

            # Assign a material if specified
            # NOTE: Only the first material is considered for now
            materials = geom_spec.materials
            if len(materials) > 0:
                descriptor.mid = material_index_map.get(str(materials[0]), 0)
            msg.info(f"[{name}]: material_index_map: {material_index_map}")
            msg.info(f"[{name}]: materials: {materials}")
            msg.info(f"[{name}]: mid: {descriptor.mid}")

            # Assign collision group/filters if specified
            collision_group_paths = geom_spec.collisionGroups
            filtered_collisions_paths = geom_spec.filteredCollisions
            msg.info(f"[{name}]: collision_group_paths: {collision_group_paths}")
            msg.info(f"[{name}]: filtered_collisions_paths: {filtered_collisions_paths}")
            collision_groups = []
            for collision_group_path in collision_group_paths:
                collision_groups.append(cgroup_index_map.get(str(collision_group_path), 0))
            geom_group = min(collision_groups) if len(collision_groups) > 0 else 1
            msg.info(f"[{name}]: collision_groups: {collision_groups}")
            msg.info(f"[{name}]: geom_group: {geom_group}")
            geom_collides = geom_group
            for cgroup in collision_groups:
                if cgroup != geom_group:
                    geom_collides += cgroup
            msg.info(f"[{name}]: geom_collides: {geom_collides}")
            descriptor.group = geom_group
            descriptor.collides = geom_collides

        # Return the final descriptor
        return descriptor

    ###
    # Public API
    ###

    def import_from(
        self,
        source: str,
        root_path: str = "/",
        xform: Transform | None = None,
        ignore_paths: list[str] | None = None,
        builder: ModelBuilder | None = None,
        apply_up_axis_from_stage: bool = True,
        only_load_enabled_rigid_bodies: bool = True,
        only_load_enabled_joints: bool = True,
        load_static_geometry: bool = True,
        load_materials: bool = True,
        enable_self_collisions: bool = True,
        collapse_fixed_joints: bool = False,
    ) -> ModelBuilder:
        """
        Parses an OpenUSD file.
        """

        # Check if the source is a valid USD file path or an existing stage
        if isinstance(source, str):
            stage = self.Usd.Stage.Open(source, self.Usd.Stage.LoadAll)
        # TODO: When does this case happen?
        else:
            stage = source

        ###
        # Units
        ###

        # Load the global distance, rotation and mass units from the stage
        rotation_unit = np.pi / 180
        distance_unit = 1.0
        mass_unit = 1.0
        try:
            if self.UsdGeom.StageHasAuthoredMetersPerUnit(stage):
                distance_unit = self.UsdGeom.GetStageMetersPerUnit(stage)
        except Exception as e:
            msg.error(f"Failed to get linear unit: {e}")
        try:
            if self.UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
                mass_unit = self.UsdPhysics.GetStageKilogramsPerUnit(stage)
        except Exception as e:
            msg.error(f"Failed to get mass unit: {e}")
        msg.info(f"distance_unit: {distance_unit}")
        msg.info(f"rotation_unit: {rotation_unit}")
        msg.info(f"mass_unit: {mass_unit}")

        ###
        # Preparation
        ###

        # Initialize the ingore paths as an empty list if it is None
        # NOTE: This is required by the LoadUsdPhysicsFromRange method
        if ignore_paths is None:
            ignore_paths = []

        # Load the USD file into an object dictionary
        ret_dict = self.UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=ignore_paths)

        # Create a new ModelBuilder if not provided
        if builder is None:
            builder = ModelBuilder()

        ###
        # World
        ###

        # Initialize the world properties
        gravity = GravityDescriptor()

        # Parse for PhysicsScene prims
        if self.UsdPhysics.ObjectType.Scene in ret_dict:
            # Retrive the phusics sene path and description
            paths, scene_descs = ret_dict[self.UsdPhysics.ObjectType.Scene]
            path, scene_desc = paths[0], scene_descs[0]
            msg.info(f"Found PhysicsScene at {path}")
            if len(paths) > 1:
                msg.error("Multiple PhysicsScene prims found in the USD file. Only the first prim will be considered.")

            # Extract the world gravity from the physics scene
            gravity = GravityDescriptor()
            gravity.acceleration = distance_unit * scene_desc.gravityMagnitude
            gravity.direction = vec3f(scene_desc.gravityDirection)
            builder.set_gravity(gravity)
            msg.info(f"World gravity: {gravity}")

            # Set the world up-axis based on the gravity direction
            up_axis = Axis.from_any(int(np.argmax(np.abs(scene_desc.gravityDirection))))
        else:
            # NOTE: Gravity is left with default values
            up_axis = Axis.from_string(str(self.UsdGeom.GetStageUpAxis(stage)))

        # Determine the up-axis transformation
        if apply_up_axis_from_stage:
            builder.set_up_axis(up_axis)
            axis_xform = wp.transform_identity()
            msg.info(f"Using stage up axis {up_axis} as builder up axis")
        else:
            axis_xform = wp.transform(vec3f(0.0), quat_between_axes(up_axis, builder.up_axis))
            msg.info(f"Rotating stage to align its up axis {up_axis} with builder up axis {builder.up_axis}")

        # Set the world offset transform based on the provided xform
        if xform is None:
            world_xform = axis_xform
        else:
            world_xform = wp.transform(*xform) * axis_xform
        msg.info(f"World offset transform: {world_xform}")

        ###
        # Materials
        ###

        # Initialize an empty materials map
        material_index_map = {}

        # Load materials only if requested
        if load_materials:
            # TODO: mechanism to detect multiple default overrides
            # Parse for and import UsdPhysicsRigidBodyMaterialDesc entries
            if self.UsdPhysics.ObjectType.RigidBodyMaterial in ret_dict:
                prim_paths, rigid_body_material_specs = ret_dict[self.UsdPhysics.ObjectType.RigidBodyMaterial]
                for prim_path, material_spec in zip(prim_paths, rigid_body_material_specs, strict=False):
                    msg.info(f"Parsing material @'{prim_path}': {material_spec}")
                    material_desc = self._parse_material(
                        material_prim=stage.GetPrimAtPath(prim_path),
                        distance_unit=distance_unit,
                        mass_unit=mass_unit,
                    )
                    if material_desc is not None:
                        has_default_override = self._get_material_default_override(stage.GetPrimAtPath(prim_path))
                        if has_default_override:
                            msg.info(f"Overriding default material with:\n{material_desc}\n")
                            builder.set_default_material(material=material_desc)
                        else:
                            msg.info(f"Adding material '{builder.num_materials}':\n{material_desc}\n")
                            material_index_map[str(prim_path)] = builder.add_material(material=material_desc)
            msg.debug(f"material_index_map: {material_index_map}")

            # Generate material pair properties for each combination
            # NOTE: This applies the OpenUSD convention of using the average of the two properties
            for i, first in enumerate(builder.materials.materials):
                for j, second in enumerate(builder.materials.materials):
                    if i <= j:  # Avoid duplicate pairs
                        msg.info(f"Generating material pair properties for '{first.name}' and '{second.name}'")
                        material_pair = self._material_pair_properties_from(first, second)
                        msg.debug(f"material_pair: {material_pair}")
                        builder.set_material_pair(
                            first=first.name,
                            second=second.name,
                            material_pair=material_pair,
                        )

        ###
        # Collision Groups
        ###

        # Parse for and import UsdPhysicsCollisionGroup prims
        cgroup_count = 0
        cgroup_index_map = {}
        if self.UsdPhysics.ObjectType.CollisionGroup in ret_dict:
            prim_paths, collision_group_specs = ret_dict[self.UsdPhysics.ObjectType.CollisionGroup]
            for prim_path, collision_group_spec in zip(prim_paths, collision_group_specs, strict=False):
                msg.info(f"Parsing collision group @'{prim_path}': {collision_group_spec}")
                cgroup_index_map[str(prim_path)] = cgroup_count + 1
                cgroup_count += 1
        msg.debug(f"cgroup_count: {cgroup_count}")
        msg.debug(f"cgroup_index_map: {cgroup_index_map}")

        ###
        # Bodies
        ###

        # Define a mapping from prim paths to body indices
        # NOTE: This can be used for both rigid and flexible bodies
        body_index_map = {}

        # Parse for and import UsdPhysicsRigidBody prims
        if self.UsdPhysics.ObjectType.RigidBody in ret_dict:
            prim_paths, rigid_body_specs = ret_dict[self.UsdPhysics.ObjectType.RigidBody]
            for prim_path, rigid_body_spec in zip(prim_paths, rigid_body_specs, strict=False):
                msg.info(f"Parsing rigid body @'{prim_path}'")
                rigid_body_desc = self._parse_rigid_body(
                    only_load_enabled_rigid_bodies=only_load_enabled_rigid_bodies,
                    rigid_body_prim=stage.GetPrimAtPath(prim_path),
                    rigid_body_spec=rigid_body_spec,
                    offset_xform=world_xform,
                    distance_unit=distance_unit,
                    rotation_unit=rotation_unit,
                    mass_unit=mass_unit,
                )
                if rigid_body_desc is not None:
                    msg.info(f"Adding body '{builder.num_bodies}':\n{rigid_body_desc}\n")
                    body_index_map[str(prim_path)] = builder.add_rigid_body_descriptor(descriptor=rigid_body_desc)
        msg.debug(f"body_index_map: {body_index_map}")

        ###
        # Joints
        ###

        # First construct lists of joint prim paths and their types tha
        # retain the order of the joints as specified in the USD file.
        joint_prim_paths = []
        joint_type_names = []
        for prim in stage.Traverse():
            if prim.GetTypeName() in self.supported_usd_joint_type_names:
                joint_type_names.append(prim.GetTypeName())
                joint_prim_paths.append(prim.GetPath())
        msg.debug(f"joint_prim_paths: {joint_prim_paths}")
        msg.debug(f"joint_type_names: {joint_type_names}")

        # Then iterate over each pair of prim path and joint type-name to parse the joint specifications
        for joint_prim_path, joint_type_name in zip(joint_prim_paths, joint_type_names, strict=False):
            joint_type = self.supported_usd_joint_types[self.supported_usd_joint_type_names.index(joint_type_name)]
            joint_paths, joint_specs = ret_dict[joint_type]
            for prim_path, joint_spec in zip(joint_paths, joint_specs, strict=False):
                if prim_path == joint_prim_path:
                    msg.info(f"Parsing joint @'{prim_path}' of type '{joint_type_name}'")
                    joint_desc = self._parse_joint(
                        only_load_enabled_joints=only_load_enabled_joints,
                        joint_prim=stage.GetPrimAtPath(prim_path),
                        joint_spec=joint_spec,
                        joint_type=joint_type,
                        body_index_map=body_index_map,
                        distance_unit=distance_unit,
                        rotation_unit=rotation_unit,
                    )
                    if joint_desc is not None:
                        msg.info(f"Adding joint '{builder.num_joints}':\n{joint_desc}\n")
                        builder.add_joint_descriptor(descriptor=joint_desc)
                    break  # Stop after the first match

        ###
        # Geometry
        ###

        # Traverse the stage to collect geometry prim paths and their types
        geom_prim_paths = []
        geom_type_names = []
        for prim in stage.Traverse():
            if prim.GetTypeName() in self.supported_usd_geom_type_names:
                geom_type_names.append(prim.GetTypeName())
                geom_prim_paths.append(prim.GetPath())
        msg.debug(f"geom_prim_paths: {geom_prim_paths}")
        msg.debug(f"geom_type_names: {geom_type_names}")

        # Construct a list of geometry layers
        # TODO: Define a mechanism to handle multiple layers
        builder.add_collision_layer(name="primary")
        builder.add_physical_layer(name="primary")
        if load_static_geometry:
            builder.add_collision_layer(name="world")
            builder.add_physical_layer(name="world")

        # Iterate over each pair of prim path and geom type-name to parse the geometry specifications
        for geom_prim_path, geom_type_name in zip(geom_prim_paths, geom_type_names, strict=False):
            geom_type = self.supported_usd_geom_types[self.supported_usd_geom_type_names.index(geom_type_name)]
            geom_paths, geom_specs = ret_dict[geom_type]
            for prim_path, geom_spec in zip(geom_paths, geom_specs, strict=False):
                if prim_path == geom_prim_path:
                    msg.info(f"Parsing geometry @'{prim_path}' of type '{geom_type_name}'")
                    geom_desc = self._parse_geom(
                        geom_prim=stage.GetPrimAtPath(prim_path),
                        geom_spec=geom_spec,
                        geom_type=geom_type,
                        body_index_map=body_index_map,
                        cgroup_index_map=cgroup_index_map,
                        material_index_map=material_index_map,
                        distance_unit=distance_unit,
                        rotation_unit=rotation_unit,
                    )
                    if geom_desc is not None:
                        # Skip static geometry if not requested
                        if geom_desc.bid == -1 and not load_static_geometry:
                            continue
                        # Append geometry descriptor to appropriate entity
                        if type(geom_desc) is CollisionGeometryDescriptor:
                            msg.info(f"Adding collision geom '{builder.num_collision_geoms}':\n{geom_desc}\n")
                            builder.add_collision_geometry_descriptor(descriptor=geom_desc)
                        elif type(geom_desc) is GeometryDescriptor:
                            msg.info(f"Adding physical geom '{builder.num_physical_geoms}':\n{geom_desc}\n")
                            builder.add_physical_geometry_descriptor(descriptor=geom_desc)
                    break  # Stop after the first match

        ###
        # Post-processing
        ###

        # TODO: enable_self_collisions
        # TODO: collapse_fixed_joints

        # if collapse_fixed_joints:
        #     collapse_results = builder.collapse_fixed_joints()
        #     body_merged_parent = collapse_results["body_merged_parent"]
        #     body_merged_transform = collapse_results["body_merged_transform"]
        #     body_remap = collapse_results["body_remap"]

        ###
        # Summary
        ###

        msg.debug(f"Builder: Rigid Bodies:\n{builder.bodies}\n")
        msg.debug(f"Builder: Joints:\n{builder.joints}\n")
        msg.debug(f"Builder: Physical Geoms:\n{builder.physical_geoms}\n")
        msg.debug(f"Builder: Collision Geoms:\n{builder.collision_geoms}\n")
        msg.debug(f"Builder: Materials:\n{builder.materials.materials}\n")

        # Return the ModelBuilder populated from the parsed USD file
        return builder
