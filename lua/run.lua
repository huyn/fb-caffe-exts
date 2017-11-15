local transfer = require 'style_transfer'

local function main()
	--local input_str = {'input_image':'1.jpg','output_image':'2.jpg','model_code':'s528','max_length':256}
	local input_str={}
	input_str['input_image']='1.jpg'
	input_str['output_image']='2.jpg'
	input_str['model_code']='s528'
	input_str['max_length']=256
	transfer.set_model_base_path('/User/huyaonan/caffetest/lua')
	transfer.transfer_single_image(input_str)
end

main()