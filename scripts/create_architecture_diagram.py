"""
Create a professional, publication-quality architecture diagram
for the Raman Spectral Decomposition model (v2).

v2 changes: mixture-only encoder, linear ref projection, softmax output.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

# ── Colors ────────────────────────────────────────────────────────────
C_INPUT     = '#4FC3F7'
C_ENCODER   = '#7E57C2'
C_REFPROJ   = '#AB47BC'
C_ATTENTION = '#FF7043'
C_FEATURES  = '#66BB6A'
C_SCORER    = '#EF5350'
C_BASELINE  = '#FFA726'
C_OUTPUT    = '#26C6DA'
C_BG        = '#FAFAFA'

fig, ax = plt.subplots(1, 1, figsize=(14, 19))
ax.set_xlim(0, 14)
ax.set_ylim(0, 19)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)


def draw_box(ax, x, y, w, h, color, text, fontsize=11, fontcolor='white',
             alpha=0.95, bold=True):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor='white',
        linewidth=1.5, alpha=alpha, zorder=3)
    box.set_path_effects([
        pe.withSimplePatchShadow(offset=(1.5, -1.5), shadow_rgbFace='#BDBDBD', alpha=0.3)])
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=fontcolor, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color='#455A64', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=15), zorder=2)


def draw_label(ax, x, y, text, fontsize=9, color='#616161'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontstyle='italic', color=color, zorder=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))


# ── Title ─────────────────────────────────────────────────────────────
ax.text(7, 18.3, 'Raman Spectral Decomposition Network (v2)',
        ha='center', va='center', fontsize=18, fontweight='bold', color='#263238')
ax.text(7, 17.8, 'Architecture Overview', ha='center', va='center', fontsize=13, color='#607D8B')

# ══════════════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 4, 17, 3.5, 0.8, C_INPUT, 'Unknown Spectrum  y', fontsize=12, fontcolor='#01579B')
draw_box(ax, 10.5, 17, 3.8, 0.8, C_INPUT, 'Reference Spectra  R₁..Rₖ', fontsize=12, fontcolor='#01579B')

draw_label(ax, 4, 16.35, '(B, 3001)', fontsize=8, color='#90A4AE')
draw_label(ax, 10.5, 16.35, '(B, K, 3001)', fontsize=8, color='#90A4AE')

# ══════════════════════════════════════════════════════════════════════
# MIXTURE ENCODER (left only — NOT shared with references!)
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 4, 15.2, 3.2, 0.65, C_ENCODER, 'Conv1D Stack (3 blocks)', fontsize=11)
draw_box(ax, 4, 14.4, 3.2, 0.65, C_ENCODER, 'Transformer (1 layer)', fontsize=10)

draw_arrow(ax, 4, 16.55, 4, 15.55)
draw_arrow(ax, 4, 14.85, 4, 14.75)

# "Mixture only" label
ax.text(1.5, 14.8, 'Mixture\nEncoder\nOnly', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#7B1FA2', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1BEE7', edgecolor='#CE93D8',
                  linewidth=1.5, alpha=0.5))

# ══════════════════════════════════════════════════════════════════════
# REFERENCE PROJECTION (right — lightweight linear, NOT deep encoder)
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 10.5, 14.8, 3.5, 0.7, C_REFPROJ, 'Linear Projection', fontsize=11)

draw_arrow(ax, 10.5, 16.55, 10.5, 15.18)

# "No deep encoder" label
ax.text(13.2, 15.6, 'No deep\nencoder!', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#C62828', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', edgecolor='#EF9A9A',
                  linewidth=1.5, alpha=0.7))

draw_label(ax, 10.5, 14.25, 'Linear(3001, d) + LayerNorm', fontsize=8, color='#6A1B9A')

# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 4, 13.3, 2.8, 0.65, '#9575CD', 'z_u  (d=128)', fontsize=11)
draw_box(ax, 10.5, 13.3, 3.2, 0.65, '#9575CD', 'z_r  (K × d=128)', fontsize=11)

draw_arrow(ax, 4, 14.05, 4, 13.65)
draw_arrow(ax, 10.5, 14.42, 10.5, 13.65)

draw_label(ax, 4, 13.85, 'Global Avg Pool', fontsize=8)

# ══════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7, 11.8, 4.5, 0.85, C_ATTENTION, 'Cross-Attention', fontsize=13)

draw_arrow(ax, 4, 12.95, 5.5, 12.25, color='#7E57C2')
draw_arrow(ax, 10.5, 12.95, 8.5, 12.25, color='#7E57C2')

draw_label(ax, 4.3, 12.55, 'Query', fontsize=9, color='#E65100')
draw_label(ax, 10.0, 12.55, 'Key, Value', fontsize=9, color='#E65100')

# ══════════════════════════════════════════════════════════════════════
# CONTEXT + z_r
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 4.5, 10.5, 2.5, 0.6, '#FF8A65', 'Context', fontsize=11)
draw_box(ax, 10, 10.5, 2.2, 0.6, '#FF8A65', 'z_rᵢ', fontsize=11)

draw_arrow(ax, 6, 11.35, 4.5, 10.85)
draw_arrow(ax, 8, 11.35, 10, 10.85)

# ══════════════════════════════════════════════════════════════════════
# SPECTRAL FEATURES
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7.2, 9.3, 4.0, 0.85, C_FEATURES, 'Spectral Features', fontsize=12)

ax.text(7.2, 8.7, 'cosine sim  |  dot product  |  L2 distance',
        ha='center', va='center', fontsize=9, fontstyle='italic', color='#2E7D32',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='none', alpha=0.9))

draw_arrow(ax, 4.5, 10.15, 5.8, 9.75, color=C_FEATURES)
draw_arrow(ax, 10, 10.15, 8.6, 9.75, color=C_FEATURES)

# Bypass arrow from inputs
ax.annotate('', xy=(5.0, 9.5), xytext=(1.5, 17),
            arrowprops=dict(arrowstyle='->', color=C_FEATURES, lw=1.5,
                            connectionstyle='arc3,rad=0.3', linestyle='--', alpha=0.6), zorder=1)
draw_label(ax, 1.8, 13.0, 'Signal-space\nbypass', fontsize=8, color='#2E7D32')

# ══════════════════════════════════════════════════════════════════════
# CONCATENATION
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7.2, 7.7, 5.5, 0.7, '#78909C',
         'Concatenate  [ z_rᵢ  |  context  |  features ]', fontsize=10, fontcolor='white')
draw_label(ax, 7.2, 7.15, '(B, K, 2d+3)', fontsize=8, color='#90A4AE')

draw_arrow(ax, 4.5, 10.15, 5.0, 8.1, color='#78909C')
draw_arrow(ax, 10, 10.15, 9.4, 8.1, color='#78909C')
draw_arrow(ax, 7.2, 8.65, 7.2, 8.08)

# ══════════════════════════════════════════════════════════════════════
# SCORER MLP
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7.2, 6.3, 4.2, 0.8, C_SCORER, 'Per-Reference Scorer MLP', fontsize=12)

ax.text(7.2, 5.7, 'Linear(2d+3, d) → ReLU → Dropout → Linear(d, 1)',
        ha='center', va='center', fontsize=8, fontstyle='italic', color='#B71C1C',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='none', alpha=0.9))

draw_arrow(ax, 7.2, 6.8, 7.2, 6.73)

# ══════════════════════════════════════════════════════════════════════
# SOFTMAX (NEW in v2!)
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7.2, 4.8, 3.0, 0.7, '#7B1FA2', 'Softmax', fontsize=13, fontcolor='white')

ax.text(11, 4.8, 'Sum-to-one\n+ Non-negative\n(architectural)', ha='center', va='center',
        fontsize=9, fontstyle='italic', color='#4A148C', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E1BEE7', edgecolor='#CE93D8',
                  linewidth=1.5, alpha=0.5))

draw_arrow(ax, 7.2, 5.4, 7.2, 5.18, color=C_SCORER)

# ══════════════════════════════════════════════════════════════════════
# OUTPUT: Coefficients
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 7.2, 3.8, 3.5, 0.65, C_OUTPUT, 'Coefficients  c₁..cₖ', fontsize=13, fontcolor='#004D40')
draw_arrow(ax, 7.2, 4.42, 7.2, 4.15, color='#7B1FA2')

# ══════════════════════════════════════════════════════════════════════
# BASELINE HEAD
# ══════════════════════════════════════════════════════════════════════
draw_box(ax, 2.5, 6.3, 3.0, 0.7, C_BASELINE, 'Baseline Head MLP', fontsize=11)

ax.text(2.5, 5.7, 'Linear(d, d) → ReLU → Linear(d, order+1)',
        ha='center', va='center', fontsize=8, fontstyle='italic', color='#E65100',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='none', alpha=0.9))

draw_box(ax, 2.5, 3.8, 3.2, 0.65, C_BASELINE, 'Baseline  b(x) ≥ 0', fontsize=12,
         fontcolor='#4E342E', alpha=0.8)

ax.annotate('', xy=(2.5, 6.68), xytext=(4, 12.95),
            arrowprops=dict(arrowstyle='->', color=C_BASELINE, lw=2.0,
                            connectionstyle='arc3,rad=0.35'), zorder=2)
draw_label(ax, 1.8, 10.0, 'z_u', fontsize=9, color='#E65100')

draw_arrow(ax, 2.5, 5.4, 2.5, 4.15, color=C_BASELINE)
draw_label(ax, 2.5, 3.25, 'Polynomial (order 5)', fontsize=8, color='#90A4AE')

# ══════════════════════════════════════════════════════════════════════
# PHYSICS DECODER (reconstruction)
# ══════════════════════════════════════════════════════════════════════
eq_box = FancyBboxPatch(
    (1.5, 1.3), 11.5, 1.4,
    boxstyle="round,pad=0.2", facecolor='white', edgecolor='#B0BEC5',
    linewidth=1.5, alpha=0.95, zorder=3)
ax.add_patch(eq_box)

ax.text(7.2, 2.35, 'Physics Decoder:    $\\hat{y}$  =  $\\sum_{i=1}^{K}$ $c_i$ $\\cdot$ $R_i$  +  b(x)',
        ha='center', va='center', fontsize=13, color='#263238', zorder=4)
ax.text(7.2, 1.75, 'Loss = $\\lambda_c$ MAE(c) + $\\lambda_r$ MSE($\\hat{y}$, y) + $\\lambda_{SAD}$ SAD($\\hat{y}$, y) + $\\lambda_b$ MAE(b) + $\\lambda_{L1}$ ||c||$_1$',
        ha='center', va='center', fontsize=10, color='#546E7A', zorder=4)

# ══════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT, 'Input spectra'),
    (C_ENCODER, 'Mixture encoder'),
    (C_REFPROJ, 'Reference projection'),
    (C_ATTENTION, 'Cross-attention'),
    (C_FEATURES, 'Spectral features'),
    (C_SCORER, 'Scorer + Softmax'),
    (C_BASELINE, 'Baseline head'),
    (C_OUTPUT, 'Output'),
]

for i, (color, label) in enumerate(legend_items):
    x = 11.8
    y = 6.2 - i * 0.38
    rect = FancyBboxPatch((x, y - 0.12), 0.35, 0.24,
                          boxstyle="round,pad=0.03",
                          facecolor=color, edgecolor='white', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + 0.55, y, label, ha='left', va='center', fontsize=9, color='#424242')

# ── Save ──
out = '/gpfs0/bgu-rgilad/users/orelgr/deep2/outputs/figs/architecture_diagram.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=C_BG, pad_inches=0.3)
plt.close()
print(f'Saved to {out}')
