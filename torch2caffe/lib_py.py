"""
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import logging

import caffe
import torch2caffe.caffe_layers
import torch2caffe.caffe_builder

from google.protobuf import text_format

log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

Layer = collections.namedtuple(
    'Layer',
    ['typename', 'name', 'index', 'torch_layer', 'bottom_edges', 'top_edges'])

Edge = collections.namedtuple(
    'Edge',
    ['name'])


class NetBuilder(object):
    def __init__(self):
        self._layers = []
        self._edges = []
        self._namecount = 0

    def _new_layer_name(self, torch_layer, typename):
        if 'name' in torch_layer:
            return torch_layer['name']
        count = self._namecount
        self._namecount += 1
        return '%s_%d' % (typename, count)

    def new_named_edges(self, edge_names):
        self._edges += [Edge(name=name) for name in edge_names]
        log.info('edge_names %s, len %d', edge_names, len(edge_names))
        return range(len(self._edges) - len(edge_names), len(self._edges))

    def _new_edges(self, layername, num=1):
        new_edges = (
            [Edge(name='%s_%d' % (layername, i)) for i in range(0, num)]
            if num > 1 else
            [Edge(name=layername)])
        self._edges += new_edges
        return range(len(self._edges) - num, len(self._edges))

    def add_layer(self,
                  typename,
                  torch_layer,
                  bottom_edges,
                  top_edges,
                  num_bottoms,
                  num_tops,
                  inplace):
        # log.info(torch_layer)
        # # if typename == 'caffe.SpatialUpSamplingNearest' or typename == 'caffe.ReLU':
        # if typename == 'caffe.SpatialUpSamplingNearest':
        #     output = torch_layer["output"]
        #     # print("print output size")
        #     # print(output[0].size[1])
        #     # print(output)
        #     print("output 0:")
        #     print(output[0])
        #     print("start print output size")
        #     print(len(output))
        #     print(len(output[0]))
        #     # print(output[0].size[1])
        #     # print(output[0].size[0])
        #     print("end print output size")
        #     assert torch_layer is None
        #
        # log.info(bottom_edges)
        # if bottom_edges is None:
        #     log.info('bottom_edge is none')
        # if bottom_edges == None:
        #     log.info('bottom_edge == None')

        num_tops = None if num_tops is None else int(num_tops)
        num_bottoms = None if num_bottoms is None else int(num_bottoms)
        bottom_edges = None if bottom_edges is None else [int(bottom_edge) for bottom_edge in bottom_edges]
        # log.info('new bottom_edges')
        # log.info(bottom_edges)
        top_edges = None if top_edges is None else [int(top_edge) for top_edge in top_edges]

        layername = self._new_layer_name(torch_layer, typename)
        assert num_bottoms is None or num_bottoms == len(bottom_edges), (
            'Invalid number of inputs for %s. Got %d, needs %d.' % (
                layername, len(bottom_edges), num_bottoms))
        assert num_tops is not None or top_edges is not None, (
            'Cannot determine number of outputs for %s' % layername)
        assert (
            num_tops is None or
            top_edges is None or
            num_tops == len(top_edges)
        ), 'Invalid number of outputs for %s. Got %d, needs %d.' % (
            layername, len(top_edges), num_tops)
        if top_edges is None:
            if not inplace:
                top_edges = self._new_edges(layername, num=num_tops)
            else:
                assert len(bottom_edges) == num_tops
                top_edges = bottom_edges

        self._layers.append(Layer(
            typename=typename,
            name=layername,
            index=self._namecount-1,
            torch_layer=torch_layer,
            bottom_edges=bottom_edges,
            top_edges=top_edges))
        log.info(
            'added layer %s: bottom=[%s], top=[%s]',
            layername,
            ','.join(self._edges[x].name for x in bottom_edges),
            ','.join(self._edges[x].name for x in top_edges))
        return top_edges

    def edge_by_name(self, name):
        for i, edge in enumerate(self._edges):
            if edge.name == name:
                return i
        return None

    @property
    def layers(self):
        return self._layers

    @property
    def edges(self):
        return self._edges


def initialize():
    return NetBuilder()


def setup_inputs(opts, net):
    return net.new_named_edges([input['name'] for input in opts['inputs']])


def setup_outputs(opts, net):
    return None if 'outputs' not in opts else (
        net.new_named_edges([output['name'] for output in opts['outputs']]))


def finalize(opts, net):
    text_net, binary_weights = torch2caffe.caffe_builder.to_caffe(
        net.layers,
        net.edges,
        opts)
    torch2caffe.caffe_builder.save(opts, text_net, binary_weights)


def print_all(obj):
    modulelist = dir(obj)
    length = len(modulelist)
    print('=================print_all...start')
    for i in range(0,length,1):
        print(modulelist[i])
    print('=================print_all...end')


def load(opts):
    net = caffe.Net(opts["prototxt"], opts["caffemodel"], caffe.TEST)
    assert net, "Net is none?"
    # for name in net._layer_names:
        # print(net.params[name])
        # if name == "ConvNdBackward1":
            # print_all(net.params[name])
            # data = net.params[name][0].data
            # bias = net.params[name][1].data
            # print(net.params[name][0].data[1][0][0][0])
            # print(net.params[name][0].data[0][1][0][0])
            # print(net.params[name][0].data)
            # print("======================")
            # print(len(net.params[name]))
            # print(len(data))
            # print(len(data[0]))
            # print(len(data[0][0]))
            # print(len(data[0][0][0]))
            # i1 = len(data)
            # i2 = len(data[0])
            # i3 = len(data[0][0])
            # i4 = len(data[0][0][0])
            # print(len(bias))
            # for j1 in range(0, i1):
            #     for j2 in range(0, i2):
            #         for j3 in range(0, i3):
            #             for j4 in range(0, i4):
            #                 net.params[name][0].data[j1][j2][j3][j4] = 1
            # for m in range(0, len(bias)):
            #     net.params[name][1].data[m] = 0
            # print("after revalue")
            # print(net.params[name][0].data)
            # print(net.params[name][1].data)
            # print(net.params[name][0].diff)
    return net


def check_layer_names(opts, expected_names):
    net = caffe.proto.caffe_pb2.NetParameter()
    with open(opts["prototxt"]) as f:
        text_format.Merge(f.read(), net)
    assert len(expected_names) == len(net.layer), (
        'Wrong net size: need %d, got %d.' % (
            len(expected_names),
            len(net.layer)))
    for i, layer in enumerate(net.layer):
        assert layer.name == expected_names[i], (
            'Wrong layer name for layer %d: need %s, got %s.' % (
                i, expected_names[i], layer.name))


def format_output(outputs):
    """
    The output is of the form {layer_name: blob}

    Return a list of blobs in _increasing_ id order of
    layer_name suffix, which maps onto how ConcatTable, etc return their
    outputs (in the multi output case)

    Consider that the layer name is of form <name>_<id> and sort by
    increasing <id>.
    """
    def _try_int(s):
        try:
            return int(s)
        except:
            return s

    items = sorted(
        ([_try_int(s) for s in k.split('_')], v) for k, v in outputs.items())
    return [v for (_, v) in items]


def debug_net(caffe_net):
    for blob_name, blob in caffe_net.blobs.iteritems():
        log.info("Blob Name: %s, %s, Sum: %s", blob_name, blob.data.shape, blob.data.sum())
        # log.info("Blob Name: %s, %s", blob_name, blob.data.sum())
        # log.info(blob.data)