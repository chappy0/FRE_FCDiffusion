# # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # import numpy as np

# # # # # # # # # # --- 1. Data Definition ---
# # # # # # # # # # The data is organized into two paths:
# # # # # # # # # # Path 1 (Solid Lines): Varying lambda_CUNet_kd
# # # # # # # # # # Path 2 (Dashed Lines): Varying lambda_FCNet_kd

# # # # # # # # # # X-axis coordinates are the numerical ratios R = lambda_CUNet_kd / lambda_FCNet_kd
# # # # # # # # # x_coords = [0.4/7, 0.8/7, 1.6/7] # Corresponds to Low, Baseline, High for solid path
# # # # # # # # # # The ratios for the dashed path are [0.8/14, 0.8/7, 0.8/3.5], which are numerically identical.

# # # # # # # # # x_labels = ['Low Ratio', 'Baseline', 'High Ratio']

# # # # # # # # # # --- Y-axis Data ---
# # # # # # # # # # Path 1 (Solid Lines) Data
# # # # # # # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # # # # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # # # # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # # # # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # # # # # # # Path 2 (Dashed Lines) Data
# # # # # # # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # # # # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # # # # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # # # # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # # # # # # --- 2. Create Figure and Axes ---
# # # # # # # # # fig, ax1 = plt.subplots(figsize=(10, 6.5)) # Create a figure and the left y-axis
# # # # # # # # # ax2 = ax1.twinx() # Create the right y-axis sharing the same x-axis

# # # # # # # # # # --- 3. Plot Data Series ---
# # # # # # # # # # Plot on the left axis (ax1)
# # # # # # # # # # We only provide a "label" for the solid lines, as they will represent the metric in the legend.
# # # # # # # # # h1 = ax1.plot(x_coords, struct_sim_solid, color='blue', linestyle='-', marker='o', label='Struct. Sim.')
# # # # # # # # # ax1.plot(x_coords, struct_sim_dashed, color='blue', linestyle='--', marker='o', markerfacecolor='white')

# # # # # # # # # h2 = ax1.plot(x_coords, text_img_sim_solid, color='red', linestyle='-', marker='s', label='Text-Img. Sim.')
# # # # # # # # # ax1.plot(x_coords, text_img_sim_dashed, color='red', linestyle='--', marker='s', markerfacecolor='white')

# # # # # # # # # h3 = ax1.plot(x_coords, lpips_solid, color='orange', linestyle='-', marker='^', label='LPIPS')
# # # # # # # # # ax1.plot(x_coords, lpips_dashed, color='orange', linestyle='--', marker='^', markerfacecolor='white')

# # # # # # # # # # Plot on the right axis (ax2)
# # # # # # # # # h4 = ax2.plot(x_coords, brisque_solid, color='green', linestyle='-', marker='D', label='BRISQUE')
# # # # # # # # # ax2.plot(x_coords, brisque_dashed, color='green', linestyle='--', marker='D', markerfacecolor='white')

# # # # # # # # # # --- 4. Configure Plot Aesthetics ---
# # # # # # # # # # Set x-axis to a logarithmic scale
# # # # # # # # # ax1.set_xscale('log')

# # # # # # # # # # Set titles and labels using LaTeX for math symbols
# # # # # # # # # ax1.set_title('Sensitivity to Distillation Loss Ratio and Path', fontsize=16)
# # # # # # # # # ax1.set_xlabel(r'Ratio $\lambda_{CUNet\_kd} / \lambda_{FCNet\_kd}$', fontsize=12)
# # # # # # # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=16)
# # # # # # # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=16)

# # # # # # # # # # Set x-axis ticks and labels
# # # # # # # # # ax1.set_xticks(x_coords)
# # # # # # # # # ax1.set_xticklabels(x_labels, fontsize=11)
# # # # # # # # # ax1.minorticks_off()

# # # # # # # # # # Set y-axis limits
# # # # # # # # # ax1.set_ylim(0, 1.05)
# # # # # # # # # ax2.set_ylim(24, 27)

# # # # # # # # # # Add grid
# # # # # # # # # ax1.grid(True, which="both", ls="--", color='gray', alpha=0.5)

# # # # # # # # # # --- 5. Create Legend ---
# # # # # # # # # # Combine handles from both axes to create a single legend
# # # # # # # # # handles = h1 + h2 + h3 + h4
# # # # # # # # # fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), frameon=False,fontsize=16)

# # # # # # # # # # --- 6. Save Figure ---
# # # # # # # # # # Adjust layout to prevent labels/legend from being cut off
# # # # # # # # # plt.tight_layout()
# # # # # # # # # fig.subplots_adjust(bottom=0.15) # Make space for the legend

# # # # # # # # # # Save the figure
# # # # # # # # # plt.savefig('sensitivity_plot.pdf')
# # # # # # # # # plt.savefig('sensitivity_plot.png', dpi=300)

# # # # # # # # # # plt.show() # Uncomment to display the plot interactively

# # # # # # # # # print("Plot saved successfully as 'sensitivity_plot.pdf' and 'sensitivity_plot.png'")


# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # import numpy as np

# # # # # # # # # --- 1. Data Definition ---
# # # # # # # # # The data is organized into two paths:
# # # # # # # # # Path 1 (Solid Lines): Varying lambda_CUNet_kd
# # # # # # # # # Path 2 (Dashed Lines): Varying lambda_FCNet_kd
# # # # # # # # x_indices = [0, 1, 2] # Use simple indices for even visual spacing
# # # # # # # # x_labels = ['Low Ratio\n(0.057)', 'Baseline\n(0.229)', 'High Ratio\n(0.457)']

# # # # # # # # # --- Y-axis Data ---
# # # # # # # # # Path 1 (Solid Lines) Data
# # # # # # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # # # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # # # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # # # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # # # # # # Path 2 (Dashed Lines) Data
# # # # # # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # # # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # # # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # # # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # # # # # --- 2. Aesthetics & Professional Color Palette ---
# # # # # # # # plt.style.use('seaborn-v0_8-whitegrid') # Use a clean, professional plot style
# # # # # # # # colors = {
# # # # # # # #     'struct': '#0072B2',  # Blue
# # # # # # # #     'text': '#D55E00',    # Vermillion
# # # # # # # #     'lpips': '#E69F00',   # Orange
# # # # # # # #     'brisque': '#009E73' # Green
# # # # # # # # }
# # # # # # # # LINE_WIDTH = 2.5
# # # # # # # # MARKER_SIZE = 8

# # # # # # # # # --- 3. Create Figure and Axes ---
# # # # # # # # fig, ax1 = plt.subplots(figsize=(12, 8)) # A slightly wider figure for better spacing
# # # # # # # # ax2 = ax1.twinx()

# # # # # # # # # --- 4. Plot Data Series with Enhanced Aesthetics ---
# # # # # # # # # Plot on the left axis (ax1)
# # # # # # # # h1 = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # h2 = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # h3 = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # # Plot on the right axis (ax2)
# # # # # # # # h4 = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # # --- 5. Configure Plot Aesthetics with Larger Fonts ---
# # # # # # # # # Set titles and labels
# # # # # # # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=22, pad=20)
# # # # # # # # ax1.set_xlabel(r'Ratio: $\lambda_{CUNet\_kd} / \lambda_{FCNet\_kd}$', fontsize=18, labelpad=15)
# # # # # # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=18, labelpad=15)
# # # # # # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=18, labelpad=15)

# # # # # # # # # Set x-axis ticks and labels for even spacing
# # # # # # # # ax1.set_xticks(x_indices)
# # # # # # # # ax1.set_xticklabels(x_labels, fontsize=14)

# # # # # # # # # Set y-axis tick label sizes
# # # # # # # # ax1.tick_params(axis='y', labelsize=14)
# # # # # # # # ax2.tick_params(axis='y', labelsize=14)

# # # # # # # # # Set y-axis limits for better focus
# # # # # # # # ax1.set_ylim(0.25, 0.95)
# # # # # # # # ax2.set_ylim(24.5, 26.5)

# # # # # # # # # --- 6. Create a More Informative Legend ---
# # # # # # # # # Create custom legend elements for the line styles
# # # # # # # # from matplotlib.lines import Line2D
# # # # # # # # solid_line = Line2D([0], [0], color='black', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$')
# # # # # # # # dashed_line = Line2D([0], [0], color='black', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$')

# # # # # # # # # Combine handles from both axes and custom lines
# # # # # # # # handles = h1 + h2 + h3 + h4
# # # # # # # # legend1 = fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=16)
# # # # # # # # legend2 = fig.legend(handles=[solid_line, dashed_line], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=14)
# # # # # # # # fig.add_artist(legend1) # Add the first legend manually

# # # # # # # # # --- 7. Save Figure ---
# # # # # # # # # Adjust layout to prevent labels/legend from being cut off
# # # # # # # # plt.tight_layout()
# # # # # # # # fig.subplots_adjust(bottom=0.2) # Make more space for the two-part legend

# # # # # # # # # Save the figure
# # # # # # # # plt.savefig('sensitivity_plot_refined.pdf')
# # # # # # # # plt.savefig('sensitivity_plot_refined.png', dpi=300)

# # # # # # # # print("Refined plot saved successfully as 'sensitivity_plot_refined.pdf' and 'sensitivity_plot_refined.png'")

# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import numpy as np

# # # # # # # # --- 1. Data Definition ---
# # # # # # # x_indices = [0, 1, 2] # Use simple indices for even visual spacing
# # # # # # # # New symbolic ratio labels as requested
# # # # # # # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # # # # # # --- Y-axis Data ---
# # # # # # # # Path 1 (Solid Lines): Varying lambda_CUNet_kd (0.5:1, 1:1, 2:1)
# # # # # # # # Path 2 (Dashed Lines): Varying lambda_FCNet_kd (1:2, 1:1, 1:0.5)
# # # # # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # # # # --- 2. Aesthetics & Professional Color Palette ---
# # # # # # # plt.style.use('seaborn-v0_8-whitegrid')
# # # # # # # colors = {
# # # # # # #     'struct': '#0072B2',  # Blue
# # # # # # #     'text': '#D55E00',    # Vermillion
# # # # # # #     'lpips': '#E69F00',   # Orange
# # # # # # #     'brisque': '#009E73' # Green
# # # # # # # }
# # # # # # # LINE_WIDTH = 3.0 # Thicker lines
# # # # # # # MARKER_SIZE = 9  # Larger markers

# # # # # # # # --- 3. Create Figure and Axes (Smaller Size) ---
# # # # # # # fig, ax1 = plt.subplots(figsize=(10, 6.5)) # Reduced figure size
# # # # # # # ax2 = ax1.twinx()

# # # # # # # # --- 4. Plot Data Series ---
# # # # # # # # Plot on the left axis (ax1)
# # # # # # # h1 = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # h2 = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # h3 = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # Plot on the right axis (ax2)
# # # # # # # h4 = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # # --- 5. Configure Plot Aesthetics (Larger Fonts & Adjusted Ticks) ---
# # # # # # # # Set titles and labels
# # # # # # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # # # # # # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=15)
# # # # # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # # # # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # # # # # # # Set x-axis ticks and labels
# # # # # # # ax1.set_xticks(x_indices)
# # # # # # # ax1.set_xticklabels(x_labels, fontsize=16)

# # # # # # # # Set y-axis limits and ticks to de-emphasize BRISQUE's variation
# # # # # # # ax1.set_ylim(0.25, 0.95)
# # # # # # # ax1.set_yticks(np.arange(0.3, 1.0, 0.15)) # Larger tick intervals
# # # # # # # ax2.set_ylim(24.0, 28.0) # Wider range to make the line look 'flatter'
# # # # # # # ax2.set_yticks(np.arange(24, 29, 1)) # Larger tick intervals

# # # # # # # # Set y-axis tick label sizes
# # # # # # # ax1.tick_params(axis='y', labelsize=16)
# # # # # # # ax2.tick_params(axis='y', labelsize=16)

# # # # # # # # --- 6. Create a More Informative Legend ---
# # # # # # # from matplotlib.lines import Line2D
# # # # # # # solid_line = Line2D([0], [0], color='black', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$')
# # # # # # # dashed_line = Line2D([0], [0], color='black', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$')

# # # # # # # handles = h1 + h2 + h3 + h4
# # # # # # # legend1 = fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=18)
# # # # # # # legend2 = fig.legend(handles=[solid_line, dashed_line], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=16)
# # # # # # # fig.add_artist(legend1)

# # # # # # # # --- 7. Save Figure ---
# # # # # # # plt.tight_layout()
# # # # # # # fig.subplots_adjust(bottom=0.2) # Adjust space for the legend

# # # # # # # plt.savefig('sensitivity_plot_final.pdf')
# # # # # # # plt.savefig('sensitivity_plot_final.png', dpi=300)

# # # # # # # print("Final plot saved successfully as 'sensitivity_plot_final.pdf' and 'sensitivity_plot_final.png'")

# # # # # # import matplotlib.pyplot as plt
# # # # # # import numpy as np

# # # # # # # --- 1. Data Definition ---
# # # # # # x_indices = [0, 1, 2]
# # # # # # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # # # # # --- Y-axis Data ---
# # # # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # # # --- 2. Aesthetics & Professional Color Palette ---
# # # # # # plt.style.use('seaborn-v0_8-whitegrid')
# # # # # # colors = {
# # # # # #     'struct': '#0072B2',
# # # # # #     'text': '#D55E00',
# # # # # #     'lpips': '#E69F00',
# # # # # #     'brisque': '#009E73'
# # # # # # }
# # # # # # LINE_WIDTH = 3.0
# # # # # # MARKER_SIZE = 9

# # # # # # # --- 3. Create Figure and Axes ---
# # # # # # fig, ax1 = plt.subplots(figsize=(10, 6.5))
# # # # # # ax2 = ax1.twinx()

# # # # # # # --- 4. Plot Data Series ---
# # # # # # h1 = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # h2 = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # h3 = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # h4 = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # # --- 5. Configure Plot Aesthetics ---
# # # # # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # # # # # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=15)
# # # # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # # # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # # # # # ax1.set_xticks(x_indices)
# # # # # # ax1.set_xticklabels(x_labels, fontsize=16)

# # # # # # ax1.set_ylim(0.25, 0.95)
# # # # # # ax1.set_yticks(np.arange(0.3, 1.0, 0.15))
# # # # # # ax2.set_ylim(24.0, 28.0)
# # # # # # ax2.set_yticks(np.arange(24, 29, 1))

# # # # # # ax1.tick_params(axis='y', labelsize=16)
# # # # # # ax2.tick_params(axis='y', labelsize=16)

# # # # # # # --- 6. Create a More Informative Legend with Adjusted Positions ---
# # # # # # from matplotlib.lines import Line2D
# # # # # # solid_line = Line2D([0], [0], color='black', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$')
# # # # # # dashed_line = Line2D([0], [0], color='black', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$')

# # # # # # handles = h1 + h2 + h3 + h4
# # # # # # # Adjusted bbox_to_anchor y-values to prevent overlap
# # # # # # legend1 = fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=18)
# # # # # # legend2 = fig.legend(handles=[solid_line, dashed_line], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=16)
# # # # # # fig.add_artist(legend1)

# # # # # # # --- 7. Save Figure with the definitive fix ---
# # # # # # # Using bbox_inches='tight' ensures all artists (like legends) are included in the final saved file.
# # # # # # # Increased padding slightly to ensure the lower legend has space.
# # # # # # plt.savefig('sensitivity_plot_final_layoutfix.pdf', bbox_inches='tight', pad_inches=0.2)
# # # # # # plt.savefig('sensitivity_plot_final_layoutfix.png', dpi=300, bbox_inches='tight', pad_inches=0.2)

# # # # # # print("Final plot with overlap fix saved successfully.")

# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np

# # # # # # --- 1. Data Definition ---
# # # # # x_indices = [0, 1, 2]
# # # # # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # # # # --- Y-axis Data ---
# # # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # # --- 2. Aesthetics & Professional Color Palette ---
# # # # # plt.style.use('seaborn-v0_8-whitegrid')
# # # # # colors = {
# # # # #     'struct': '#0072B2',
# # # # #     'text': '#D55E00',
# # # # #     'lpips': '#E69F00',
# # # # #     'brisque': '#009E73'
# # # # # }
# # # # # LINE_WIDTH = 3.0
# # # # # MARKER_SIZE = 9

# # # # # # --- 3. Create Figure and Axes ---
# # # # # fig, ax1 = plt.subplots(figsize=(10, 6.5))
# # # # # ax2 = ax1.twinx()

# # # # # # --- 4. Plot Data Series ---
# # # # # h1 = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # h2 = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # h3 = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # h4 = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # # --- 5. Configure Plot Aesthetics ---
# # # # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # # # # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=15)
# # # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # # # # ax1.set_xticks(x_indices)
# # # # # ax1.set_xticklabels(x_labels, fontsize=16)

# # # # # ax1.set_ylim(0.25, 0.95)
# # # # # ax1.set_yticks(np.arange(0.3, 1.0, 0.15))
# # # # # ax2.set_ylim(24.0, 28.0)
# # # # # ax2.set_yticks(np.arange(24, 29, 1))

# # # # # ax1.tick_params(axis='y', labelsize=16)
# # # # # ax2.tick_params(axis='y', labelsize=16)

# # # # # # --- 6. Create a Single, Unified Legend ---
# # # # # from matplotlib.lines import Line2D
# # # # # # Get the handles for the data series
# # # # # data_handles = h1 + h2 + h3 + h4
# # # # # # Create "proxy artists" for the line styles
# # # # # style_handles = [
# # # # #     Line2D([0], [0], color='black', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$ (Solid)'),
# # # # #     Line2D([0], [0], color='black', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
# # # # # ]
# # # # # # Combine all handles
# # # # # all_handles = data_handles + style_handles

# # # # # # Create a single legend with 3 columns, which will arrange the 6 items into two rows
# # # # # fig.legend(handles=all_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=16)

# # # # # # --- 7. Save Figure ---
# # # # # # Using bbox_inches='tight' ensures all artists are included in the final saved file.
# # # # # plt.savefig('sensitivity_plot_final_layoutfix.pdf', bbox_inches='tight', pad_inches=0.1)
# # # # # plt.savefig('sensitivity_plot_final_layoutfix.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# # # # # print("Final plot with unified legend saved successfully.")

# # # # import matplotlib.pyplot as plt
# # # # import numpy as np

# # # # # --- 1. Data Definition ---
# # # # x_indices = [0, 1, 2]
# # # # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # # # --- Y-axis Data ---
# # # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # # --- 2. Aesthetics & Professional Color Palette ---
# # # # plt.style.use('seaborn-v0_8-whitegrid')
# # # # colors = {
# # # #     'struct': '#0072B2',
# # # #     'text': '#D55E00',
# # # #     'lpips': '#E69F00',
# # # #     'brisque': '#009E73'
# # # # }
# # # # LINE_WIDTH = 3.0
# # # # MARKER_SIZE = 9

# # # # # --- 3. Create Figure and Axes ---
# # # # fig, ax1 = plt.subplots(figsize=(10.5, 6)) # Adjusted figsize for better aspect ratio
# # # # ax2 = ax1.twinx()

# # # # # --- 4. Plot Data Series ---
# # # # # Note: We now add labels directly here for the axis legend
# # # # h1, = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # h2, = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # h3, = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # h4, = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # # --- 5. Configure Plot Aesthetics ---
# # # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # # # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=10)
# # # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # # # ax1.set_xticks(x_indices)
# # # # ax1.set_xticklabels(x_labels, fontsize=16)

# # # # ax1.set_ylim(0.25, 0.95)
# # # # ax1.set_yticks(np.arange(0.3, 1.0, 0.15))
# # # # ax2.set_ylim(24.0, 28.0)
# # # # ax2.set_yticks(np.arange(24, 29, 1))

# # # # ax1.tick_params(axis='y', labelsize=16)
# # # # ax2.tick_params(axis='y', labelsize=16)

# # # # # --- 6. Create a Legend Inside the Plot ---
# # # # from matplotlib.lines import Line2D
# # # # # We need to get handles from both axes to show all metrics
# # # # handles1, labels1 = ax1.get_legend_handles_labels()
# # # # handles2, labels2 = ax2.get_legend_handles_labels()
# # # # all_handles = handles1 + handles2

# # # # # Create proxy artists for the line styles
# # # # style_handles = [
# # # #     Line2D([0], [0], color='gray', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$ (Solid)'),
# # # #     Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
# # # # ]

# # # # # Create the legend inside the plot area
# # # # legend = ax1.legend(handles=all_handles + style_handles, 
# # # #                     loc='upper right',       # Position the legend
# # # #                     ncol=1,                  # Arrange in a single column
# # # #                     fontsize=14,             # Set font size
# # # #                     bbox_to_anchor=(1, 0.98), # Fine-tune position
# # # #                     frameon=True,            # Add a frame for clarity
# # # #                     facecolor='white',       # Set background color
# # # #                     edgecolor='black',       # Set frame color
# # # #                     framealpha=0.8)          # Make background semi-transparent

# # # # # --- 7. Save Figure ---
# # # # plt.savefig('sensitivity_plot_final_legend_inside.pdf', bbox_inches='tight')
# # # # plt.savefig('sensitivity_plot_final_legend_inside.png', dpi=300, bbox_inches='tight')

# # # # print("Final plot with an internal legend saved successfully.")


# # # import matplotlib.pyplot as plt
# # # import numpy as np

# # # # --- 1. Data Definition ---
# # # x_indices = [0, 1, 2]
# # # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # # --- Y-axis Data ---
# # # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # # lpips_solid =         [0.5063, 0.493, 0.5065]
# # # brisque_solid =       [25.5268, 26.32, 24.8479]

# # # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # # --- 2. Aesthetics & Professional Color Palette ---
# # # plt.style.use('seaborn-v0_8-whitegrid')
# # # colors = {
# # #     'struct': '#0072B2',
# # #     'text': '#D55E00',
# # #     'lpips': '#E69F00',
# # #     'brisque': '#009E73'
# # # }
# # # LINE_WIDTH = 3.0
# # # MARKER_SIZE = 9

# # # # --- 3. Create Figure and Axes ---
# # # fig, ax1 = plt.subplots(figsize=(10.5, 6.5)) # Slightly increased height for bottom space
# # # ax2 = ax1.twinx()

# # # # --- 4. Plot Data Series ---
# # # h1, = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # h2, = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # h3, = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # h4, = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # # --- 5. Configure Plot Aesthetics ---
# # # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # # # Reduced labelpad to pull the x-label up, away from the legend
# # # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=10)
# # # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # # ax1.set_xticks(x_indices)
# # # ax1.set_xticklabels(x_labels, fontsize=16)

# # # ax1.set_ylim(0.25, 0.95)
# # # ax1.set_yticks(np.arange(0.3, 1.0, 0.15))
# # # ax2.set_ylim(24.0, 28.0)
# # # ax2.set_yticks(np.arange(24, 29, 1))

# # # ax1.tick_params(axis='y', labelsize=16)
# # # ax2.tick_params(axis='y', labelsize=16)

# # # # --- 6. Create a Single, Unified Legend at the Bottom ---
# # # from matplotlib.lines import Line2D
# # # handles1, labels1 = ax1.get_legend_handles_labels()
# # # handles2, labels2 = ax2.get_legend_handles_labels()
# # # all_handles = handles1 + handles2

# # # style_handles = [
# # #     Line2D([0], [0], color='gray', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$ (Solid)'),
# # #     Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
# # # ]
# # # # Combine all handles
# # # all_handles = all_handles + style_handles

# # # # Create a single legend, positioned far below the x-axis label
# # # fig.legend(handles=all_handles,
# # #            loc='lower center',
# # #            ncol=3,
# # #            # Significantly lowered the legend's vertical position
# # #            bbox_to_anchor=(0.5, -0.3),
# # #            frameon=False,
# # #            fontsize=16)

# # # # --- 7. Save Figure ---
# # # # Increased pad_inches to ensure the lowered legend is saved
# # # plt.savefig('sensitivity_plot.pdf', bbox_inches='tight', pad_inches=0.3)
# # # plt.savefig('sensitivity_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

# # # print("Final plot with bottom legend and no overlap saved successfully.")


# # import matplotlib.pyplot as plt
# # import numpy as np

# # # --- 1. Data Definition ---
# # x_indices = [0, 1, 2]
# # x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # # --- Y-axis Data ---
# # struct_sim_solid =    [0.8922, 0.902, 0.8948]
# # text_img_sim_solid =  [0.305,  0.316, 0.3069]
# # lpips_solid =         [0.5063, 0.493, 0.5065]
# # brisque_solid =       [25.5268, 26.32, 24.8479]

# # struct_sim_dashed =   [0.899,  0.902, 0.887]
# # text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# # lpips_dashed =        [0.4959, 0.493, 0.5328]
# # brisque_dashed =      [25.4489, 26.32, 26.2158]

# # # --- 2. Aesthetics & Professional Color Palette ---
# # plt.style.use('seaborn-v0_8-whitegrid')
# # colors = {
# #     'struct': '#0072B2',
# #     'text': '#D55E00',
# #     'lpips': '#E69F00',
# #     'brisque': '#009E73'
# # }
# # LINE_WIDTH = 3.0
# # MARKER_SIZE = 9

# # # --- 3. Create Figure and Axes (Reduced Height) ---
# # # Reduced the height from 6.5 to 6.0 to make the plot more compact
# # fig, ax1 = plt.subplots(figsize=(10.5, 6.0))
# # ax2 = ax1.twinx()

# # # --- 4. Plot Data Series ---
# # h1, = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # h2, = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # h3, = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # h4, = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# # ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # # --- 5. Configure Plot Aesthetics ---
# # ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# # ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=10)
# # ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# # ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# # ax1.set_xticks(x_indices)
# # ax1.set_xticklabels(x_labels, fontsize=16)

# # ax1.set_ylim(0.25, 0.95)
# # # Decreased y-tick step for finer-grained intervals
# # ax1.set_yticks(np.arange(0.3, 1.0, 0.1))
# # ax2.set_ylim(24.0, 28.0)
# # # Decreased y-tick step for finer-grained intervals
# # ax2.set_yticks(np.arange(24, 28.5, 0.5))

# # ax1.tick_params(axis='y', labelsize=16)
# # ax2.tick_params(axis='y', labelsize=16)

# # # --- 6. Create a Single, Unified Legend at the Bottom ---
# # from matplotlib.lines import Line2D
# # handles1, labels1 = ax1.get_legend_handles_labels()
# # handles2, labels2 = ax2.get_legend_handles_labels()
# # all_handles = handles1 + handles2

# # style_handles = [
# #     Line2D([0], [0], color='gray', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$ (Solid)'),
# #     Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
# # ]
# # all_handles = all_handles + style_handles

# # fig.legend(handles=all_handles,
# #            loc='lower center',
# #            ncol=3,
# #            bbox_to_anchor=(0.5, -0.3),
# #            frameon=False,
# #            fontsize=16)

# # # --- 7. Save Figure ---
# # plt.savefig('sensitivity_plot.pdf', bbox_inches='tight', pad_inches=0.3)
# # plt.savefig('sensitivity_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

# # print("Publication-ready plot saved successfully.")

# import matplotlib.pyplot as plt
# import numpy as np

# # --- 1. Data Definition ---
# x_indices = [0, 1, 2]
# x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# # --- Y-axis Data ---
# struct_sim_solid =    [0.8922, 0.902, 0.8948]
# text_img_sim_solid =  [0.305,  0.316, 0.3069]
# lpips_solid =         [0.5063, 0.493, 0.5065]
# brisque_solid =       [25.5268, 26.32, 24.8479]

# struct_sim_dashed =   [0.899,  0.902, 0.887]
# text_img_sim_dashed = [0.3015, 0.316, 0.3019]
# lpips_dashed =        [0.4959, 0.493, 0.5328]
# brisque_dashed =      [25.4489, 26.32, 26.2158]

# # --- 2. Aesthetics & Professional Color Palette ---
# plt.style.use('seaborn-v0_8-whitegrid')
# colors = {
#     'struct': '#0072B2',
#     'text': '#D55E00',
#     'lpips': '#E69F00',
#     'brisque': '#009E73'
# }
# LINE_WIDTH = 3.0
# MARKER_SIZE = 9

# # --- 3. Create Figure and Axes (More Compact Height) ---
# # Reduced the height significantly from 6.5 to 5.5 to make the plot more compact
# fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
# ax2 = ax1.twinx()

# # --- 4. Plot Data Series ---
# h1, = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# h2, = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# h3, = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# h4, = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
# ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE)

# # --- 5. Configure Plot Aesthetics ---
# ax1.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
# ax1.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=10)
# ax1.set_ylabel(r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', fontsize=20, labelpad=15)
# ax2.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# ax1.set_xticks(x_indices)
# ax1.set_xticklabels(x_labels, fontsize=16)

# # Tightly set y-axis limits to reduce empty space
# ax1.set_ylim(0.28, 0.95)
# ax1.set_yticks(np.arange(0.30, 0.91, 0.15)) # Ticks for the tightened range

# # Set right axis to end at 28
# ax2.set_ylim(24.5, 28)
# ax2.set_yticks(np.arange(25, 28.1, 1))

# ax1.tick_params(axis='y', labelsize=16)
# ax2.tick_params(axis='y', labelsize=16)

# # --- 6. Create a Single, Unified Legend at the Bottom ---
# from matplotlib.lines import Line2D
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# all_handles = handles1 + handles2

# style_handles = [
#     Line2D([0], [0], color='gray', lw=LINE_WIDTH, label=r'Varying $\lambda_{CUNet\_kd}$ (Solid)'),
#     Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
# ]
# all_handles = all_handles + style_handles

# # Adjusted y-anchor for the more compact figure
# fig.legend(handles=all_handles,
#            loc='lower center',
#            ncol=3,
#            bbox_to_anchor=(0.5, -0.35),
#            frameon=False,
#            fontsize=16)

# # --- 7. Save Figure ---
# # Increased pad_inches to ensure the legend is saved in the new compact layout
# plt.savefig('sensitivity_plot.pdf', bbox_inches='tight', pad_inches=0.4)
# plt.savefig('sensitivity_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.4)

# print("Compact, publication-ready plot saved successfully.")


import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Definition ---
x_indices = [0, 1, 2]
x_labels = ['0.5:1 / 1:2', '1:1\n(Baseline)', '2:1 / 1:0.5']

# --- Y-axis Data ---
struct_sim_solid =    [0.8922, 0.902, 0.8948]
text_img_sim_solid =  [0.305,  0.316, 0.3069]
lpips_solid =         [0.5063, 0.493, 0.5065]
brisque_solid =       [25.5268, 26.32, 24.8479]

struct_sim_dashed =   [0.899,  0.902, 0.887]
text_img_sim_dashed = [0.3015, 0.316, 0.3019]
lpips_dashed =        [0.4959, 0.493, 0.5328]
brisque_dashed =      [25.4489, 26.32, 26.2158]

# --- 2. Aesthetics & Professional Color Palette ---
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'struct': '#0072B2',
    'text': '#D55E00',
    'lpips': '#E69F00',
    'brisque': '#009E73'
}
LINE_WIDTH = 3.0
MARKER_SIZE = 9

# --- 3. Create Figure and Subplots for the "Broken Axis" ---
# Create two subplots, one for the top part and one for the bottom
# `sharex=True` links their x-axes. `height_ratios` makes the bottom part taller.
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10.5, 6), 
                                       gridspec_kw={'height_ratios': [1, 2.5]})
fig.subplots_adjust(hspace=0.1)  # Adjust space between the two subplots

# Create twin axes for the BRISQUE score for both subplots
ax2_top = ax_top.twinx()
ax2_bottom = ax_bottom.twinx()

# --- 4. Plot Data on BOTH sets of axes ---
axes_pairs = [(ax_top, ax2_top), (ax_bottom, ax2_bottom)]
for ax1, ax2 in axes_pairs:
    h1, = ax1.plot(x_indices, struct_sim_solid, color=colors['struct'], linestyle='-', marker='o', label='Struct. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)
    ax1.plot(x_indices, struct_sim_dashed, color=colors['struct'], linestyle='--', marker='o', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)

    h2, = ax1.plot(x_indices, text_img_sim_solid, color=colors['text'], linestyle='-', marker='s', label='Text-Img. Sim.', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)
    ax1.plot(x_indices, text_img_sim_dashed, color=colors['text'], linestyle='--', marker='s', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)

    h3, = ax1.plot(x_indices, lpips_solid, color=colors['lpips'], linestyle='-', marker='^', label='LPIPS', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)
    ax1.plot(x_indices, lpips_dashed, color=colors['lpips'], linestyle='--', marker='^', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)

    h4, = ax2.plot(x_indices, brisque_solid, color=colors['brisque'], linestyle='-', marker='D', label='BRISQUE', linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)
    ax2.plot(x_indices, brisque_dashed, color=colors['brisque'], linestyle='--', marker='D', markerfacecolor='white', markeredgewidth=1.5, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, clip_on=False)

# --- 5. Set Y-Limits to Create the "Break" ---
ax_top.set_ylim(0.88, 0.92)
ax_bottom.set_ylim(0.28, 0.58)

ax2_top.set_ylim(27.5, 28) # Corresponding BRISQUE range for the top plot
ax2_bottom.set_ylim(24.5, 27) # Corresponding BRISQUE range for the bottom plot

# --- 6. Configure Spines, Ticks, and Labels ---
# Hide the connecting spines and x-axis ticks for the top plot
ax_top.spines['bottom'].set_visible(False)
ax2_top.spines['bottom'].set_visible(False)
ax_bottom.spines['top'].set_visible(False)
ax2_bottom.spines['top'].set_visible(False)
ax_top.tick_params(axis='x', which='both', bottom=False)

# Add the "break" slashes
d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)      # top-left diagonal
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # bottom-right diagonal

# Set titles and labels
ax_top.set_title('Sensitivity to Distillation Loss Ratio', fontsize=24, pad=20)
ax_bottom.set_xlabel(r'Ratio ($\lambda_{CUNet\_kd} : \lambda_{FCNet\_kd}$)', fontsize=20, labelpad=10)

# Set a single, shared Y-label for the left side
fig.text(0.04, 0.5, r'Metric Score (Sim $\uparrow$, LPIPS $\downarrow$)', va='center', rotation='vertical', fontsize=20)
ax2_bottom.set_ylabel(r'BRISQUE Score $\downarrow$', fontsize=20, labelpad=15)

# Configure ticks
ax_bottom.set_xticks(x_indices)
ax_bottom.set_xticklabels(x_labels, fontsize=16)
ax_top.tick_params(axis='y', labelsize=16)
ax_bottom.tick_params(axis='y', labelsize=16)
ax2_top.tick_params(axis='y', labelsize=16)
ax2_bottom.tick_params(axis='y', labelsize=16)

# --- 7. Create Unified Legend ---
from matplotlib.lines import Line2D
all_handles = [h1, h2, h3, h4]
style_handles = [
    Line2D([0], [0], color='gray', lw=LINE_WIDTH, label='Varying $\lambda_{CUNet\_kd}$ (Solid)'),
    Line2D([0], [0], color='gray', linestyle='--', lw=LINE_WIDTH, label=r'Varying $\lambda_{FCNet\_kd}$ (Dashed)')
]
fig.legend(handles=all_handles + style_handles,
           loc='lower center',
           ncol=3,
           bbox_to_anchor=(0.5, -0.3),
           frameon=False,
           fontsize=16)

# --- 8. Save Figure ---
plt.savefig('sensitivity_plot.pdf', bbox_inches='tight', pad_inches=0.3)
plt.savefig('sensitivity_plot.png', dpi=300, bbox_inches='tight', pad_inches=0.3)

print("Final plot with broken axis saved successfully.")