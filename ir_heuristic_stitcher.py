     1 """
     2 Unified toolkit with Alerts: CSRNet processing + Stitching + Headcount Alerts
     3 
     4 This version includes:
     5 - 'process' subcommand to generate density maps from images using CSRNet.
     6 - 'stitch' subcommand to combine density maps and check against crowd count thresholds.
     7 """
     8 
     9 import os
    10 import sys
    11 import json
    12 from pathlib import Path
    13 import argparse
    14 import numpy as np
    15 from PIL import Image
    16 import matplotlib.pyplot as plt
    17 
    18 import torch
    19 import torch.nn as nn
    20 from torchvision import transforms
    21 
    22 # ---------------------- CSRNet Model ----------------------
    23 class CSRNet(nn.Module):
    24     def __init__(self):
    25         super(CSRNet, self).__init__()
    26         self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    27         self.backend_feat = [512, 512, 512, 256, 128, 64]
    28         self.frontend = self._make_layers(self.frontend_feat)
    29         self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
    30         self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
    31 
    32     def forward(self, x):
    33         x = self.frontend(x)
    34         x = self.backend(x)
    35         x = self.output_layer(x)
    36         return x
    37 
    38     def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
    39         d_rate = 2 if dilation else 1
    40         layers = []
    41         for v in cfg:
    42             if v == 'M':
    43                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    44             else:
    45                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
    46                 if batch_norm:
    47                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    48                 else:
    49                     layers += [conv2d, nn.ReLU(inplace=True)]
    50                 in_channels = v
    51         return nn.Sequential(*layers)
    52 
    53 # ---------------------- Utilities ----------------------
    54 
    55 def load_weights_robust(model, weights_path, map_location='cpu'):
    56     """Load weights robustly: handle dict or wrapped state dicts."""
    57     data = torch.load(weights_path, map_location=map_location)
    58     if isinstance(data, dict):
    59         possible_keys = ['state_dict', 'model', 'net', 'model_state_dict']
    60         sd = None
    61         for k in possible_keys:
    62             if k in data:
    63                 sd = data[k]
    64                 break
    65         if sd is None:
    66             sd = data
    67     else:
    68         sd = data
    69 
    70     def _clean_state_dict(sd):
    71         new_sd = {}
    72         for k, v in sd.items():
    73             new_key = k.replace('module.', '')
    74             new_sd[new_key] = v
    75         return new_sd
    76 
    77     if not isinstance(sd, dict):
    78         raise ValueError(f"Loaded weights ({weights_path}) do not appear to be a state dict.")
    79     try:
    80         model.load_state_dict(sd)
    81     except RuntimeError:
    82         model.load_state_dict(_clean_state_dict(sd))
    83     return True
    84 
    85 # ---------------------- Image processing ----------------------
    86 
    87 def _resize_keep_aspect(img, min_dim=512):
    88     """Resize PIL.Image so that smallest side >= min_dim while keeping aspect ratio."""
    89     w, h = img.size
    90     if min(w, h) >= min_dim:
    91         return img
    92     if w < h:
    93         new_w = min_dim
    94         new_h = int(h * (min_dim / w))
    95     else:
    96         new_h = min_dim
    97         new_w = int(w * (min_dim / h))
    98     return img.resize((new_w, new_h), Image.BICUBIC)
    99 
   100 
   101 def process_image(image_path, weights_path, out_prefix=None, save_visual=True, device='cpu'):
   102     """Process single image, run CSRNet, save raw density .npy and optional visualizations."""
   103     image_path = Path(image_path)
   104     if out_prefix is None:
   105         out_prefix = image_path.stem
   106     weights_path = Path(weights_path)
   107 
   108     model = CSRNet().to(device)
   109     load_weights_robust(model, str(weights_path), map_location=device)
   110     model.eval()
   111 
   112     img = Image.open(image_path).convert('RGB')
   113     img_resized = _resize_keep_aspect(img, min_dim=640)
   114 
   115     preprocess = transforms.Compose([
   116         transforms.ToTensor(),
   117         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   118     ])
   119     img_tensor = preprocess(img_resized).unsqueeze(0).to(device)
   120 
   121     with torch.no_grad():
   122         density_map_tensor = model(img_tensor)
   123     density_map = density_map_tensor.squeeze().cpu().numpy()
   124 
   125     out_h, out_w = density_map.shape
   126     in_w, in_h = img_resized.size
   127     if (out_h != in_h) or (out_w != in_w):
   128         density_map_resized = np.array(Image.fromarray(density_map).resize((in_w, in_h), resample=Image.BICUBIC))
   129     else:
   130         density_map_resized = density_map
   131 
   132     estimated_count = float(density_map_resized.sum())
   133     npy_path = Path(f"raw_density_{out_prefix}.npy")
   134     np.save(str(npy_path), density_map_resized.astype(np.float32))
   135 
   136     if save_visual:
   137         plt.figure(figsize=(12, 8))
   138         plt.imshow(img_resized)
   139         plt.imshow(density_map_resized, cmap='jet', alpha=0.5)
   140         plt.title(f"Combined Heatmap\nEstimated Count: {estimated_count:.2f}")
   141         plt.axis('off')
   142         plt.savefig(f"heatmap_{out_prefix}.png", bbox_inches='tight')
   143         plt.close()
   144 
   145     return {'npy': str(npy_path), 'estimated_count': estimated_count}
   146 
   147 
   148 def batch_process_images(image_paths, weights_path, out_prefixes=None, save_visual=True, device='cpu'):
   149     results = []
   150     if out_prefixes is None:
   151         out_prefixes = [None] * len(image_paths)
   152     for img_p, pre in zip(image_paths, out_prefixes):
   153         res = process_image(img_p, weights_path, out_prefix=pre, save_visual=save_visual, device=device)
   154         results.append(res)
   155     return results
   156 
   157 # ---------------------- Stitching ----------------------
   158 
   159 def stitch_density_maps(map_npy_paths, homographies=None, canvas_size=None, placements=None, out_image='REAL_master_heatmap.png'):
   160     """Stitch multiple density maps onto a master canvas."""
   161     maps = [np.load(p).astype(np.float32) for p in map_npy_paths]
   162     hs = [m.shape[0] for m in maps]
   163     ws = [m.shape[1] for m in maps]
   164 
   165     if canvas_size is None:
   166         if placements is not None:
   167             max_h = max(y + h for (x, y), h in zip(placements, hs))
   168             max_w = max(x + w for (x, w), w in zip(placements, ws))
   169             canvas_h, canvas_w = max_h + 50, max_w + 50
   170         else:
   171             canvas_h = max(hs) + 100
   172             canvas_w = int(sum(ws) * 0.75) + 200
   173     else:
   174         canvas_h, canvas_w = canvas_size
   175 
   176     master = np.zeros((canvas_h, canvas_w), dtype=np.float32)
   177 
   178     if homographies is not None:
   179         for m, H_path in zip(maps, homographies):
   180             H = np.load(H_path)
   181             warped = cv2.warpPerspective(m, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
   182             master += warped
   183     else:
   184         # Fallback to simple placement
   185         x_cursor = 50
   186         for m in maps:
   187             h, w = m.shape
   188             master[50:50+h, x_cursor:x_cursor+w] += m
   189             x_cursor += int(w * 0.75)
   190 
   191     # Visualization
   192     vis = master.copy()
   193     if np.all(vis == 0):
   194         norm = vis
   195     else:
   196         norm = (vis - vis.min()) / (vis.max() - vis.min())
   197     norm_uint8 = (norm * 255).astype(np.uint8)
   198     cmap = plt.get_cmap('jet')
   199     cmap_img = cmap(norm_uint8/255.0)[:, :, :3]
   200     cmap_img_uint8 = (cmap_img * 255).astype(np.uint8)
   201     Image.fromarray(cmap_img_uint8).save(out_image)
   202 
   203     return master
   204 
   205 # ---------------------- CLI Interface ----------------------
   206 
   207 def main_cli():
   208     parser = argparse.ArgumentParser(description='CSRNet toolkit: process images and stitch density maps with alerts.')
   209     sub = parser.add_subparsers(dest='cmd', required=True)
   210 
   211     # --- Subparser: process ---
   212     p_proc = sub.add_parser('process', help='Process one or more images with CSRNet')
   213     p_proc.add_argument('--weights', required=True, help='Path to CSRNet weights file (.pth)')
   214     p_proc.add_argument('images', nargs='+', help='One or more image paths')
   215     p_proc.add_argument('--out-prefixes', nargs='*', help='Optional prefixes for outputs, one per image')
   216     p_proc.add_argument('--no-visual', action='store_true', help='Do not save visual PNGs')
   217 
   218     # --- Subparser: stitch ---
   219     p_st = sub.add_parser('stitch', help='Stitch multiple .npy density maps and check alerts')
   220     p_st.add_argument('--maps', nargs='+', required=True, help='Paths to raw_density_*.npy files')
   221     p_st.add_argument('--homographies', nargs='*', help='Paths to homography .npy files')
   222     p_st.add_argument('--placements', nargs='*', help='List of "x,y" placements')
   223     p_st.add_argument('--canvas-size', help='Optional canvas size as HEIGHT,WIDTH')
   224     p_st.add_argument('--out', default='REAL_master_heatmap.png', help='Output colored image path')
   225     # --- NEW ARGUMENTS FOR ALERTS ---
   226     p_st.add_argument('--warning-threshold', type=float, help='Crowd count to trigger a WARNING.')
   227     p_st.add_argument('--alert-threshold', type=float, help='Crowd count to trigger an ALERT.')
   228 
   229     args = parser.parse_args()
   230 
   231     if args.cmd == 'process':
   232         out_prefixes = args.out_prefixes if args.out_prefixes else [None] * len(args.images)
   233         results = batch_process_images(args.images, args.weights, out_prefixes=out_prefixes, save_visual=not args.no_visual)
   234         print(json.dumps(results, indent=2))
   235 
   236     elif args.cmd == 'stitch':
   237         placements = [tuple(map(int, p.split(','))) for p in args.placements] if args.placements else None
   238         canvas_size = tuple(map(int, args.canvas_size.split(','))) if args.canvas_size else None
   239 
   240         master_density = stitch_density_maps(
   241             args.maps,
   242             homographies=args.homographies,
   243             canvas_size=canvas_size,
   244             placements=placements,
   245             out_image=args.out
   246         )
   247 
   248         # --- NEW LOGIC FOR ALERTS ---
   249         total_count = master_density.sum()
   250         print(f"\n--- CROWD ANALYSIS ---")
   251         print(f"📊 Total Estimated Crowd Count: {total_count:.2f}")
   252 
   253         if args.alert_threshold is not None and total_count >= args.alert_threshold:
   254             print(f"🚨 [ALERT] Crowd count ({total_count:.1f}) has exceeded the ALERT threshold ({args.alert_threshold}).")
   255         elif args.warning_threshold is not None and total_count >= args.warning_threshold:
   256             print(f"⚠️ [WARNING] Crowd count ({total_count:.1f}) has exceeded the WARNING threshold ({args.warning_threshold}).")
   257         else:
   258             print("✅ [SAFE] Crowd count is within safe limits.")
   259         print("----------------------\n")
   260 
   261     else:
   262         parser.print_help()
   263 
   264 if __name__ == '__main__':
   265     main_cli()

