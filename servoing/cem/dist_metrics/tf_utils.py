import threading
import tensorflow as tf

class PersistentOpEvaluator(object):
    """Evaluate a fixed TensorFlow graph repeatedly, safely, efficiently.
    Extend this class to create a particular kind of op evaluator, like an
    image encoder. In `initialize_graph`, create an appropriate TensorFlow
    graph with placeholder inputs. In `run`, evaluate this graph and
    return its result. This class will manage a singleton graph and
    session to preserve memory usage, and will ensure that this graph and
    session do not interfere with other concurrent sessions.
    A subclass of this class offers a threadsafe, highly parallel Python
    entry point for evaluating a particular TensorFlow graph.
    Example usage:
        class FluxCapacitanceEvaluator(PersistentOpEvaluator):
            \"\"\"Compute the flux capacitance required for a system.
            Arguments:
                x: Available power input, as a `float`, in jigawatts.
            Returns:
                A `float`, in nanofarads.
            \"\"\"
            def initialize_graph(self):
                self._placeholder = tf.placeholder(some_dtype)
                self._op = some_op(self._placeholder)
            def run(self, x):
                return self._op.eval(feed_dict: {self._placeholder: x})
        evaluate_flux_capacitance = FluxCapacitanceEvaluator()
        for x in xs:
            evaluate_flux_capacitance(x)

    Taken from here:
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/util.py
    """

    def __init__(self):
        super(PersistentOpEvaluator, self).__init__()
        self._session = None
        self._initialization_lock = threading.Lock()

    def _lazily_initialize(self):
        """Initialize the graph and session, if this has not yet been done."""
        with self._initialization_lock:
            if self._session:
                return
            graph = tf.Graph()
            with graph.as_default():
                self.initialize_graph()
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            self._session = tf.Session(graph=graph, config=config)

    def initialize_graph(self):
        """Create the TensorFlow graph needed to compute this operation.
        This should write ops to the default graph and return `None`.
        """
        raise NotImplementedError('Subclasses must implement "initialize_graph".')

    def run(self, *args, **kwargs):
        """Evaluate the ops with the given input.
        When this function is called, the default session will have the
        graph defined by a previous call to `initialize_graph`. This
        function should evaluate any ops necessary to compute the result of
        the query for the given *args and **kwargs, likely returning the
        result of a call to `some_op.eval(...)`.
        """
        raise NotImplementedError('Subclasses must implement "run".')

    def __call__(self, *args, **kwargs):
        self._lazily_initialize()
        with self._session.as_default():
            return self.run(*args, **kwargs)