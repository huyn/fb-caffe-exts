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

import logging
import os

# import caffe
import caffe.proto.caffe_pb2 as pb2
import numpy as np


# import sys
def print_all(obj):
    modulelist = dir(obj)
    length = len(modulelist)
    print('=================print_all...start')
    for i in range(0,length,1):
        print(modulelist[i])
    print('=================print_all...end')


log = logging.getLogger(__name__)


def as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob
    # return caffe.io.array_to_blobproto(array)


def ty(caffe_type):
    def f(_):
        layer = pb2.LayerParameter()
        layer.type = caffe_type
        return layer
    return f


def slice(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Slice"
    layer.slice_param.axis = int(torch_layer["axis"])
    layer.slice_param.slice_point.extend(
        range(1, int(torch_layer["num_slices"])))
    return layer


def embed(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Embed"
    layer.embed_param.num_output = int(torch_layer["num_output"])
    # one-based -> zero-based indexing
    layer.embed_param.input_dim = int(torch_layer["input_dim"]) + 1
    layer.embed_param.bias_term = False
    weight = torch_layer["weight"]
    assert weight.shape[0] + 1 == layer.embed_param.input_dim
    assert weight.shape[1] == layer.embed_param.num_output
    offset_weight = np.vstack([
        np.zeros(shape=(1, weight.shape[1])),
        weight
    ])
    assert offset_weight.shape[0] == layer.embed_param.input_dim
    layer.blobs.extend([as_blob(offset_weight)])
    return layer


def reduction(torch_layer):
    layer = pb2.LayerParameter()
    op = {
        "SUM": pb2.ReductionParameter.SUM,
        "MEAN": pb2.ReductionParameter.MEAN,
        "LOGSUMEXP": pb2.ReductionParameter.LOGSUMEXP,
    }[torch_layer["operation"]]
    layer.type = "Reduction"
    layer.reduction_param.operation = op
    layer.reduction_param.axis = int(torch_layer["axis"])
    return layer


def inner_product(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "InnerProduct"
    num_output = int(torch_layer["num_output"])
    weight = torch_layer["weight"]
    bias = torch_layer["bias"]
    layer.inner_product_param.num_output = num_output

    # Last layer.
    if "bias" in torch_layer:
        bias = torch_layer["bias"]
        layer.blobs.extend([as_blob(weight), as_blob(bias)])
    else:
        layer.inner_product_param.axis = -1
        layer.blobs.extend([as_blob(weight), as_blob(bias)])
    return layer


def concat(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Concat"
    layer.concat_param.axis = int(torch_layer["axis"])
    log.info("Concat on axis %s", layer.concat_param.axis)
    return layer


def temporal_convolution(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Convolution"
    weight = torch_layer["weight"]
    bias = torch_layer["bias"]
    layer.convolution_param.num_output = int(torch_layer["outputFrameSize"])
    layer.convolution_param.kernel_w = int(torch_layer["inputFrameSize"])
    layer.convolution_param.stride_w = 1
    layer.convolution_param.kernel_h = int(torch_layer["kW"])
    layer.convolution_param.stride_h = int(torch_layer["dW"])

    blob_weight = weight.reshape(
        layer.convolution_param.num_output,
        1,
        layer.convolution_param.kernel_h,
        layer.convolution_param.kernel_w)
    layer.blobs.extend([as_blob(blob_weight), as_blob(bias)])
    return layer


def spatial_convolution(torch_layer):
    # log.info("do spatial deconvolution")
    # log.info(torch_layer)
    layer = pb2.LayerParameter()
    layer.type = "Convolution"
    bias = torch_layer["bias"]
    weight = torch_layer["weight"]
    assert len(weight.shape) == 4, weight.shape
    (nOutputPlane, nInputPlane, kH_, kW_) = weight.shape

    (kW, kH, dW, dH, padW, padH) = [
        int(torch_layer.get(f, 0))
        for f in ["kW", "kH", "dW", "dH", "padW", "padH"]]
    assert kH_ == kH
    assert kW_ == kW
    layer.convolution_param.num_output = nOutputPlane
    layer.convolution_param.kernel_w = kW
    layer.convolution_param.stride_w = dW
    layer.convolution_param.pad_w = padW
    layer.convolution_param.kernel_h = kH
    layer.convolution_param.stride_h = dH
    layer.convolution_param.pad_h = padH

    if "bias" in torch_layer:
        bias = torch_layer["bias"]
        layer.blobs.extend([as_blob(weight), as_blob(bias)])
    else:
        layer.convolution_param.bias_term = False
        layer.blobs.extend([as_blob(weight), as_blob(bias)])
    return layer


def deconvolution(torch_layer):
    log.info("do deconvolution")
    # log.info(torch_layer)
    # print_all(torch_layer["output"])
    # print_all(torch_layer["gradInput"])

    # get output and cal num_output, need forward torch
    output = torch_layer["output"]
    size = len(output)
    assert size > 0
    # print(len(output[0]))

    # gradinput = torch_layer["gradInput"]
    # print(len(output))
    # print(len(gradinput))
    # print(output[0].size[1])
    # print(gradinput[0].size[1])
    # assert torch_layer is None

    layer = pb2.LayerParameter()
    layer.type = "Deconvolution"
    factor = int(torch_layer["scale_factor"])
    layer.convolution_param.num_output = len(output[0])
    layer.convolution_param.group = len(output[0])
    # layer.convolution_param.stride = factor
    # layer.convolution_param.kernel_size = (2 * factor - factor % 2)
    layer.convolution_param.stride_w = factor
    layer.convolution_param.stride_h = factor
    layer.convolution_param.kernel_w = (2 * factor - factor % 2)
    layer.convolution_param.kernel_h = (2 * factor - factor % 2)
    layer.convolution_param.pad.append(int(np.ceil((factor - 1) / 2.)))
    layer.convolution_param.bias_term = False
    layer.convolution_param.weight_filler.type = 'bilinear'

    param_spec = pb2.ParamSpec()
    param_spec.lr_mult = 0
    param_spec.decay_mult = 0
    layer.param.extend([param_spec])

    # weight = torch_layer["weight"]
    # bias = torch_layer["bias"]
    # assert len(weight.shape) == 4, weight.shape
    # (nOutputPlane, nInputPlane, kH_, kW_) = weight.shape
    #
    # (kW, kH, dW, dH, padW, padH) = [
    #     int(torch_layer.get(f, 0))
    #     for f in ["kW", "kH", "dW", "dH", "padW", "padH"]]
    # assert kH_ == kH
    # assert kW_ == kW
    # layer.convolution_param.num_output = nOutputPlane
    # layer.convolution_param.kernel_w = kW
    # layer.convolution_param.stride_w = dW
    # layer.convolution_param.pad_w = padW
    # layer.convolution_param.kernel_h = kH
    # layer.convolution_param.stride_h = dH
    # layer.convolution_param.pad_h = padH
    #
    # if "bias" in torch_layer:
    #     bias = torch_layer["bias"]
    #     layer.blobs.extend([as_blob(weight), as_blob(bias)])
    # else:
    #     layer.convolution_param.bias_term = False
    #     layer.blobs.extend([as_blob(weight), as_blob(bias)])
    return layer


def pooling(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"
    pool = {
        "MAX": pb2.PoolingParameter.MAX,
        "AVE": pb2.PoolingParameter.AVE
    }[torch_layer["operation"]]
    layer.pooling_param.pool = pool
    (kW, kH, dW, dH, padW, padH) = [
        int(torch_layer.get(f, 0))
        for f in ["kW", "kH", "dW", "dH", "padW", "padH"]]
    layer.pooling_param.pad_w = padW
    layer.pooling_param.pad_h = padH
    layer.pooling_param.kernel_h = kH
    layer.pooling_param.kernel_w = kW
    layer.pooling_param.stride_h = dH
    layer.pooling_param.stride_w = dW
    # Default to torch_pooling, but override with the ceil_mode
    if "ceil_mode" not in torch_layer:
        return layer

    if not torch_layer["ceil_mode"]:
        # layer.pooling_param.torch_pooling = True
        if dH > 1 and padH > 0:
            layer.pooling_param.pad_h = padH - 1
        if dW > 1 and padW > 0:
            layer.pooling_param.pad_w = padW - 1
    return layer


def dropout(torch_layer):
    # Only run dropout
    layer = pb2.LayerParameter()
    layer.type = "Dropout"
    layer.dropout_param.dropout_ratio = torch_layer["p"]
    assert torch_layer["v2"], "Only handle nn.Dropout v2"
    train_only = pb2.NetStateRule()
    train_only.phase = pb2.TEST
    layer.exclude.extend([train_only])
    return layer


def fbthreshold(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "FBThreshold"
    layer.fbthreshold_param.threshold = torch_layer["threshold"]
    layer.fbthreshold_param.val = torch_layer["val"]
    return layer


def lstm_style(torch_layer):
    has_cmul = any(param.endswith("nn.CMul") for param
                   in torch_layer["layers"].keys())
    return pb2.RecurrentParameter.GRAVES if has_cmul \
        else pb2.RecurrentParameter.ZAREMBA


def get_lstm_blobs(torch_layer):
    IH_LINEAR = "_w_ih_nn.Linear"
    HH_LINEAR = "_w_hh_nn.Linear"
    CH_CMUL = "_w_ch_nn.CMul"
    layers = torch_layer["layers"]

    def filtered(suffix):
        return {
            k.replace(suffix, ""): v for k, v in layers.iteritems()
            if k.endswith(suffix)}

    ih_linear = filtered(IH_LINEAR)
    hh_linear = filtered(HH_LINEAR)
    ch_cmul = filtered(CH_CMUL)
    gate_order = ["in_gate", "forget_gate", "out_gate", "cell_gate"]
    zaremba = [
        np.vstack([ih_linear[g]["weight"] for g in gate_order]),
        np.hstack([hh_linear[g]["bias"].flatten() + ih_linear[g]["bias"].flatten()
                   for g in gate_order]),
        np.vstack([hh_linear[g]["weight"] for g in gate_order]),
    ]
    if not ch_cmul:
        assert lstm_style(torch_layer) == pb2.RecurrentParameter.ZAREMBA
        return zaremba
    assert lstm_style(torch_layer) == pb2.RecurrentParameter.GRAVES
    graves = [ch_cmul["forget_gate"]["weight"],
              ch_cmul["in_gate"]["weight"],
              ch_cmul["out_gate"]["weight"]]
    return zaremba + graves


def lstm(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "LSTM"

    lstm_blobs = get_lstm_blobs(torch_layer)
    for blob in lstm_blobs:
        log.info("Blob shape: %s", blob.shape)
    layer.recurrent_param.num_output = int(lstm_blobs[0].shape[0] / 4)
    layer.recurrent_param.lstm_style = lstm_style(torch_layer)
    if os.environ.get("T2C_DEBUG"):
        layer.recurrent_param.debug_info = True
    layer.blobs.extend([as_blob(b) for b in lstm_blobs])
    return layer


def softmax(opts):
    softmax_ty = opts["softmax"] if opts.get("softmax") else "Softmax"
    assert softmax_ty in ["Softmax", "FBSoftmax"], opts
    return ty(softmax_ty)


def batchnorm(torch_layer):
    log.info('do batchnorm')
    # log.info(torch_layer)
    layer = pb2.LayerParameter()
    layer.type = "BatchNorm"
    # Caffe BN doesn't support bias
    # assert torch_layer["affine"]==0 #would cause crash
    affine = torch_layer['affine']
    if affine:
        log.info('batch affine is true')
    else:
        log.info('batch affine is false')
    layer.batch_norm_param.use_global_stats = 1
    blobs_weight = np.ones(1)
    layer.blobs.extend([as_blob(torch_layer["running_mean"]),
                        as_blob(torch_layer["running_var"]), as_blob(blobs_weight)])
    return layer


def batchnorm_scale(torch_layer):
    layer = pb2.LayerParameter()
    layer.type = "Scale"
    layer.scale_param.bias_term = True
    layer.blobs.extend([as_blob(torch_layer["weight"]),
                        as_blob(torch_layer["bias"])])
    return layer


def leaky(torch_layer):
    log.info('print LeakyReLU')
    log.info(torch_layer)
    layer = pb2.LayerParameter()
    layer.type = "ReLU"
    layer.relu_param.negative_slope = float(torch_layer["negval"])
    return layer


def build_converter(opts):
    return {
        'caffe.Concat': concat,
        'caffe.Embed': embed,
        'caffe.Exp': ty('Exp'),
        'caffe.InnerProduct': inner_product,
        'caffe.Log': ty('Log'),
        'caffe.LogSoftmax': ty('LogSoftmax'),
        'caffe.ReLU': ty('ReLU'),
        'caffe.LeakyReLU': leaky,
        'caffe.Reduction': reduction,
        'caffe.Slice': slice,
        'caffe.Softmax': softmax(opts),
        'caffe.TanH': ty('TanH'),
        'caffe.TemporalConvolution': temporal_convolution,
        'caffe.SpatialConvolution': spatial_convolution,
        'caffe.Pooling': pooling,
        'caffe.Dropout': dropout,
        'caffe.Flatten': ty('Flatten'),
        'caffe.FBThreshold': fbthreshold,
        'caffe.LSTM': lstm,
        'caffe.BatchNorm': batchnorm,
        'caffe.BatchNormScale': batchnorm_scale,
        'caffe.SpatialUpSamplingNearest': deconvolution,
        'caffe.SpatialUpSamplingBilinear': deconvolution,
    }


def convert(opts, typename, torch_layer):
    converter = build_converter(opts)
    if typename not in converter:
        raise ValueError("Unknown layer type: {}, known types: {}".format(
            typename, converter.keys()))
    try:
        return converter[typename](torch_layer)
    except:
        log.exception("Exception on converting %s, %s", typename, torch_layer)
        raise
