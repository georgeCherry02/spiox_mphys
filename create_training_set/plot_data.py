import matplotlib.pyplot as plt

def output(tce_id, lc_views, cent_views, output_dir):
    # Plot lc_views
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, len(lc_views["global"])), lc_views["global"], 'C0.', alpha=0.5, ms=4)
    ax2 = ax1.twiny()
    ax2.plot(range(0, len(lc_views["local"])), lc_views["local"], 'C1.', alpha=0.5, ms=4)
    plt.savefig(output_dir+tce_id+"-binned_pdc.png")
    fig, ax1 = plt.subplots()
    ax1.plot(range(0, len(cent_views["global"])), cent_views["global"], 'C0.', alpha=0.5, ms=4)
    ax2 = ax1.twiny()
    ax2.plot(range(0, len(cent_views["local"])), cent_views["local"], 'C1.', alpha=0.5, ms=4)
    plt.savefig(output_dir+tce_id+"-binned_centroid.png")
