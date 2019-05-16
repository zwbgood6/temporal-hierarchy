"""Simple example on how to log scalars and images to tensorboard without tensor ops.

License: Copyleft
"""
__author__ = "Michael Gygli"

import tensorflow as tf

try:
	from StringIO import StringIO
except ImportError:
    from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from viz.html_utils.ffmpeg_gif import encode_gif


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, dataset_spec):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        
        try:
            if dataset_spec.rescale_size == "0..1":
                self.scale = lambda x: x
            else:
              # Rescale tensor from [-1, 1] to [0, 1]
                self.scale = lambda x: x / 2.0 + 0.5
        except:
            self.scale = lambda x: x

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            if img.ndim == 2:
                plt.imsave(s, img, format='png', cmap='gray')
            else:
                plt.imsave(s, self.scale(img), format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_gifs(self, tag, gif_images, step):
        """Logs list of input image vectors (nx[time x w x h x c]) into GIFs."""
        def gen_gif_summary(gif_images):
            img_list = np.split(gif_images, gif_images.shape[0], axis=0)
            enc_gif = encode_gif([i[0] for i in img_list], fps=3)
            thwc = gif_images.shape
            im_summ = tf.Summary.Image()
            im_summ.height = thwc[1]
            im_summ.width = thwc[2]
            im_summ.colorspace = thwc[3]  # fix to 3 == RGB
            im_summ.encoded_image_string = enc_gif
            return im_summ

        gif_summaries = []
        for nr, img_stack in enumerate(gif_images):
            gif_summ = gen_gif_summary(img_stack)
            # Create a Summary value
            gif_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=gif_summ))
        # Create and write Summary
        summary = tf.Summary(value=gif_summaries)
        self.writer.add_summary(summary, step)

    def log_figures(self, tag, figures, step):
        """Logs a list of figure handles."""

        fig_summaries = []
        for nr, fig in enumerate(figures):
            size = fig.get_size_inches()*fig.dpi
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            fig.savefig(s, format='png')

            # Create an Image object
            fig_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=int(size[0]),
                                       width=int(size[1]))
            # Create a Summary value
            fig_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=fig_sum))

        # Create and write Summary
        summary = tf.Summary(value=fig_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_summary(self, summary,step):
        self.writer.add_summary(summary,step)
        self.writer.flush()
