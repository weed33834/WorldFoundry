"""MoVerse Gaussian Generation Module.

Converts equirectangular panoramas + depth maps into 3D Gaussian Splatting (3DGS) PLY files.

Pipeline:
    1. DA3 (Depth Anything 3) estimates panoramic depth from the panorama image.
    2. Sharp (PanoGaussianPredictor) generates 3D Gaussians from the panorama + depth.
    3. Output is a standard 3DGS .ply file ready for real-time rendering.
"""
