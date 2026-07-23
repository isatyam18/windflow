import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from windflow import inference_flows

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Choose region bounds: [lat_min, lat_max, lon_min, lon_max]
# For full disk, use: bounds=None
# For inference, use a larger region or full disk; for rendering, use the smaller target area.
inference_bounds = [-50, 50, 20, 130]
render_bounds = [-40, 40, 30, 130]
resolution = 0.04  # Use native resolution for the selected region

def subset_region(data, bounds):
    if bounds is None:
        return data
    lat_min, lat_max, lon_min, lon_max = bounds
    lat_slice = slice(lat_min, lat_max) if data.lat.values[0] < data.lat.values[-1] else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if data.lon.values[0] < data.lon.values[-1] else slice(lon_max, lon_min)
    return data.sel(lat=lat_slice, lon=lon_slice)

# Instantiate inference runner
checkpoint_file = BASE_DIR / "model_weights" / "windflow.raft.pth.tar"
inference = inference_flows.INSATFlows(
    model_name='raft',
    overlap=128,
    tile_size=512,
    device=torch.device('cpu'),
    batch_size=4,
)
inference.load_checkpoint(checkpoint_file)

# Run inference
print("Running INSAT transfer inference...")
if inference_bounds is not None:
    print(f"Inference bounds: lat {inference_bounds[0:2]}, lon {inference_bounds[2:4]}")
else:
    print("Inference bounds: FULL DISK")
if render_bounds is not None:
    print(f"Render bounds: lat {render_bounds[0:2]}, lon {render_bounds[2:4]}")
else:
    print("Render bounds: FULL DISK")
print(f"Resolution: {resolution}°")
result = inference.flows_by_files(
    file1=str(BASE_DIR / "data" / "3SIMG_10JUN2026_0030_L1B_STD_V01R00.h5"),
    file2=str(BASE_DIR / "data" / "3SIMG_10JUN2026_0100_L1B_STD_V01R00.h5"),
    reproject=True,
    resolution=resolution,
    bounds=inference_bounds,
)
plot_result = subset_region(result, render_bounds)

print("Inference complete!")
print("Result keys:", list(result.keys()))
print("U shape:", result['U'].shape)
print("V shape:", result['V'].shape)
print("U range (m/s):", float(result['U'].min()), "to", float(result['U'].max()))
print("V range (m/s):", float(result['V'].min()), "to", float(result['V'].max()))

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs = axs.flatten()

speed = (result['U']**2 + result['V']**2)**0.5
axs[0].imshow(result['Rad'].values, cmap='gray')
axs[0].set_title("Input frame 1 (Brightness Temperature K)")
axs[1].imshow(speed.values)
axs[1].set_title("Wind Speed (m/s)")
axs[2].quiver(result['lon'].values[::20], result['lat'].values[::20], 
              result['U'].values[::20, ::20], result['V'].values[::20, ::20])
axs[2].set_title("Wind Vectors")

plt.tight_layout()
plt.savefig("insat-flows.png", dpi=150)
print("Saved plot to insat-flows.png")

# Product-style wind field visualization over the WV background with coastlines
plot_result = subset_region(result, render_bounds)
speed = np.hypot(plot_result['U'].values, plot_result['V'].values)
fig = plt.figure(figsize=(12, 8))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent([plot_result['lon'].values[0], plot_result['lon'].values[-1], plot_result['lat'].values[0], plot_result['lat'].values[-1]], crs=proj)
ax.imshow(
    plot_result['Rad'].values,
    cmap='gray_r',
    origin='lower',
    extent=[plot_result['lon'].values[0], plot_result['lon'].values[-1], plot_result['lat'].values[0], plot_result['lat'].values[-1]],
    transform=proj,
)

# Plot around 100 barbs
total_points = plot_result['U'].size
target_arrows = 1500
aqstep = max(1, int(np.sqrt(total_points / target_arrows)))
lon_mesh, lat_mesh = np.meshgrid(plot_result['lon'].values, plot_result['lat'].values)
q = ax.barbs(
    lon_mesh[::aqstep, ::aqstep],
    lat_mesh[::aqstep, ::aqstep],
    plot_result['U'].values[::aqstep, ::aqstep],
    plot_result['V'].values[::aqstep, ::aqstep],
    speed[::aqstep, ::aqstep],
    length=4,
    pivot='middle',
    linewidth=0.6,
    cmap='turbo',
    clim=(0, np.nanpercentile(speed, 98)),
    alpha=0.95,
    transform=proj,
)

ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='orange', linewidth=1)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='orange', linewidth=0.8, linestyle=':')
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='none', edgecolor='orange', linewidth=0.5, alpha=0.6)

ax.set_title('WV Background with Sparse AMV-style Wind Barbs')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Wind speed (m/s)')
plt.tight_layout()
plt.savefig("insat-flows-product.png", dpi=150)
print("Saved product-style plot to insat-flows-product.png")
