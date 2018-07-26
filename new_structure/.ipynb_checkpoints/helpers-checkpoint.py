import functools
import numpy as np
import tensorflow as tf


def lazy_property(function):
    """
    Decorator makes sure nodes are only appended if they dont already exist
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


from IPython.display import clear_output, Image, display, HTML


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def expand_array_dims(array):
    #new_array = [np.expand_dims(np.array(x), 0) for x in array]
    new_array = array.astype(np.float32).reshape((len(array), 1))

    #new_array = [np.expand_dims(x,1) for x in new_array]

    return new_array



def unison_shuffled_copies(a, b, expand_dims=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    sorted_index = np.argsort(p)
    if expand_dims == True:
        return expand_array_dims(a[p]), expand_array_dims(b[p]), sorted_index
    else:
        p = np.squeeze(p)
        return a[p], b[p], sorted_index
