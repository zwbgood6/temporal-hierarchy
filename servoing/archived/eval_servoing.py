import numpy as np

from eval import plot_time_seqs_shaded_var


_ERROR_FILES = ["/home/karl/Downloads/efficacy_vis_servoing/ours-1sh/servo_errors.npy",
                "/home/karl/Downloads/efficacy_vis_servoing/ours-10sh/servo_errors.npy",
                "/home/karl/Downloads/efficacy_vis_servoing/ours_01sh/servo_errors.npy"]
_GRAPH_LABELS = ["1sh",
                 "10sh",
                 "100sh"]

# load servoing errors
errors = [np.squeeze(np.load(f)) for f in _ERROR_FILES]
abs_errors = [np.absolute(error) for error in errors]

# print mean + stddev
for i, error in enumerate(abs_errors):
  final_error = error[:, -1]
  mean = np.mean(final_error)
  std = np.std(final_error)
  print("%s: %f +- %f" % (_GRAPH_LABELS[i], mean, std))

# plot shaded error over time
transp_abs_errors = [np.transpose(ae) for ae in abs_errors]
plot_time_seqs_shaded_var(transp_abs_errors, _GRAPH_LABELS, "Timestep", "Servoing Error")



