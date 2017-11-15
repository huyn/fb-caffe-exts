require 'torch'
require 'NoiseFill'


local util = {}


--[[
--获取文件名
--/path/to/file_name.ext return file_name
-- ]]
function util.get_file_basename(local_filepath)
    local ext = paths.extname(local_filepath)
    local basename = paths.basename(local_filepath, ext)
    return basename
end


--[[
--直方图匹配
--return image
-- ]]

function util.color_match(content_image, target_image)
    local N1, N2, C1, C2, H1, H2, W1, W2, F, Y, output_image = nil

    if content_image:dim() == 3 then
        C1, H1, W1 = content_image:size(1), content_image:size(2), content_image:size(3)
        F = content_image:view(C1, H1 * W1)
    else
        N1, C1, H1, W1 = content_image:size(1), content_image:size(2), content_image:size(3), content_image:size(4)
        F = content_image:view(C1, H1 * W1)
    end

    if target_image:dim() == 3 then
        C2, H2, W2 = target_image:size(1), target_image:size(2), target_image:size(3)
        Y = target_image:view(C2, H2 * W2)
    else
        N2, C2, H2, W2 = target_image:size(1), target_image:size(2), target_image:size(3), target_image:size(4)
        Y = target_image:view(C2, H2 * W2)
    end

    local mean_F = torch.mean(F, 2)
    local mean_Y = torch.mean(Y, 2)

    local ones_vector_F = torch.ones(1, H1 * W1):float()
    local ones_vector_Y = torch.ones(1, H2 * W2):float()

    local mean_F_matrix = torch.mm(mean_F, ones_vector_F)
    local mean_Y_matrix = torch.mm(mean_Y, ones_vector_Y)

    local M_F = torch.mm((F - mean_F_matrix), (F - mean_F_matrix):t()) / (H1 * W1)
    local M_Y = torch.mm((Y - mean_Y_matrix), (Y - mean_Y_matrix):t()) / (H2 * W2)

    local E_1e7 = torch.mul(torch.eye(M_F:size(1), M_F:size(2)), 1e-7):float()
    local L_F = torch.potrf(torch.add(M_F, E_1e7), 'L')
    local L_Y = torch.potrf(torch.add(M_Y, E_1e7), 'L')

    local beta = torch.trtrs(L_Y:t(), L_F:t(), 'U', 'N', 'N'):t()

    output_image = torch.mm(beta, F - mean_F_matrix) + torch.mm(mean_Y, ones_vector_F)

    if content_image:dim() == 3 then
        output_image = output_image:view(C1, H1, W1)
    else
        output_image = output_image:view(N1, C1, H1, W1)
    end

    return output_image
end



function util.file_exists(local_filepath)
    local file = io.open(local_filepath, "rb")
    if file then file:close() end
    return file ~= nil
end


--[[判断是否是t7模型文件
--return true | false
-- ]]
function util.is_t7_file(local_filepath)
    if string.sub(local_filepath, 1, 1) == '.' then
        return false
    end
    local ext = string.lower(paths.extname(local_filepath) or "")
    if ext == 't7' then
        return true
    else
        return false
    end
end

--[[判断是否是图片文件
--return true | false
-- ]]
local IMAGE_EXTS = { 'jpg', 'jpeg', 'png', 'ppm', 'pgm' }
function util.is_image_file(local_filepath)
    -- Hidden file are not images
    if string.sub(local_filepath, 1, 1) == '.' then
        return false
    end
    -- Check against a list of known image extensions
    local ext = string.lower(paths.extname(local_filepath) or "")
    for _, image_ext in ipairs(IMAGE_EXTS) do
        if ext == image_ext then
            return true
        end
    end
    return false
end

function util.original_colors(content, generated)
    local generated_y = image.rgb2yuv(generated)[{ { 1, 1 } }]
    local content_uv = image.rgb2yuv(content)[{ { 2, 3 } }]
    return image.yuv2rgb(torch.cat(generated_y, content_uv, 1))
end

return util