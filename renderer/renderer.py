"""
Headless mesh renderer using pyrender for DA:I character faces.

Renders extracted meshes with pyrender's offscreen EGL/OSMesa backend.
"""

import numpy as np
import trimesh
import pyrender
import os

# Force offscreen rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def load_mesh_from_arrays(positions: np.ndarray, indices: np.ndarray,
                          normals: np.ndarray = None, uvs: np.ndarray = None) -> trimesh.Trimesh:
    """Create a trimesh from extracted vertex arrays."""
    mesh = trimesh.Trimesh(
        vertices=positions,
        faces=indices,
        vertex_normals=normals,
        process=False,
    )
    if uvs is not None:
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return mesh


def load_mesh_from_npy(asset_dir: str) -> trimesh.Trimesh:
    """Load a mesh from saved numpy arrays."""
    positions = np.load(f'{asset_dir}/base_head_positions.npy')
    indices = np.load(f'{asset_dir}/base_head_indices.npy')
    normals = None
    uvs = None
    normals_path = f'{asset_dir}/base_head_normals.npy'
    uvs_path = f'{asset_dir}/base_head_uvs.npy'
    if os.path.exists(normals_path):
        normals = np.load(normals_path)
    if os.path.exists(uvs_path):
        uvs = np.load(uvs_path)
    return load_mesh_from_arrays(positions, indices, normals, uvs)


def render_face(mesh: trimesh.Trimesh, width: int = 512, height: int = 512,
                yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Render a face mesh from a given angle.

    Args:
        mesh: trimesh.Trimesh of the face
        width, height: image dimensions
        yaw: horizontal rotation in degrees (0 = front)
        pitch: vertical rotation in degrees (0 = level)

    Returns:
        RGBA image as (H, W, 4) uint8 numpy array
    """
    scene = pyrender.Scene(
        bg_color=[0.2, 0.2, 0.2, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )

    # Create pyrender mesh with a simple material
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.7, 0.65, 1.0],  # skin tone
        metallicFactor=0.0,
        roughnessFactor=0.7,
    )
    py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene.add(py_mesh)

    # Calculate camera position based on mesh bounds
    bounds = mesh.bounds  # (2, 3) array: min, max
    center = (bounds[0] + bounds[1]) / 2
    extent = (bounds[1] - bounds[0]).max()
    camera_distance = extent * 2.5

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.radians(30), aspectRatio=width / height)

    # Camera pose: look at center from front
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    # Calculate camera position
    cam_x = center[0] + camera_distance * np.sin(yaw_rad) * np.cos(pitch_rad)
    cam_y = center[1] + camera_distance * np.sin(pitch_rad)
    cam_z = center[2] + camera_distance * np.cos(yaw_rad) * np.cos(pitch_rad)

    # Build look-at matrix
    camera_pose = _look_at(
        eye=np.array([cam_x, cam_y, cam_z]),
        target=center,
        up=np.array([0.0, 1.0, 0.0]),
    )
    scene.add(camera, pose=camera_pose)

    # Key light (from upper right)
    key_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=3.0)
    key_pose = _look_at(
        eye=center + np.array([0.5, 0.5, 1.0]) * extent,
        target=center,
        up=np.array([0.0, 1.0, 0.0]),
    )
    scene.add(key_light, pose=key_pose)

    # Fill light (from left)
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.5)
    fill_pose = _look_at(
        eye=center + np.array([-0.5, 0.2, 0.8]) * extent,
        target=center,
        up=np.array([0.0, 1.0, 0.0]),
    )
    scene.add(fill_light, pose=fill_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 look-at camera matrix (OpenGL convention)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[0, :3] = right
    mat[1, :3] = true_up
    mat[2, :3] = -forward
    mat[0, 3] = eye[0]
    mat[1, 3] = eye[1]
    mat[2, 3] = eye[2]

    # pyrender expects column-major camera pose
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye

    return pose


if __name__ == '__main__':
    import sys
    from PIL import Image

    asset_dir = sys.argv[1] if len(sys.argv) > 1 else 'assets/dai'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/dai_face_render.png'

    print(f'Loading mesh from {asset_dir}...')
    mesh = load_mesh_from_npy(asset_dir)
    print(f'  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}')

    print(f'Rendering...')
    image = render_face(mesh, width=512, height=512)
    Image.fromarray(image).save(output_path)
    print(f'Saved to {output_path}')
