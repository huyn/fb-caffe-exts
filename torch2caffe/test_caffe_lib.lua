--[[
Copyright (c) 2015-present, Facebook, Inc.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
--]]
local M = {}

require 'nn'

local pl = require 'pl.import_into'()
local py = require 'fb.python'
require 'fbtorch'
local t2c = py.import('torch2caffe.lib_py')

function M.main(opts)
    --logging.infof("Opts: %s", pl.pretty.write(opts))
    print(("Opts: %s"):format(pl.pretty.write(opts)))

    local caffe_net = t2c.load(opts)
    print("compare...3")

end

return M
