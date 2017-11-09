--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local tcl = require 'torch2caffe.test_caffe_lib'
local pl = require('pl.import_into')()

local opt = pl.lapp[[
   --prototxt (default "") Output prototxt model file
   --caffemodel (default "") Output model weights file
]]

tcl.main(opt)
