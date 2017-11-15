local M = {}
local MSG_CODE = {}
M.CODE_SUCCESS = '00'
M.CODE_PARAMS_NOT_JSON = '10'
M.CODE_PARAMS_MISSING = '11'
M.CODE_MODEL_CODE_ERROR = '12'
M.CODE_FILE_NOT_EXISTS = '13'
M.CODE_UNSPPORT_TYPE = '14'
M.CODE_SYSTEM_ERROR = '19'

MSG_CODE[M.CODE_SUCCESS] = 'success'
MSG_CODE[M.CODE_PARAMS_NOT_JSON] = 'params not json'
MSG_CODE[M.CODE_PARAMS_MISSING] = 'require params missing: %s'
MSG_CODE[M.CODE_MODEL_CODE_ERROR] = 'model code error'
MSG_CODE[M.CODE_FILE_NOT_EXISTS] = 'file not exists: %s'
MSG_CODE[M.CODE_UNSPPORT_TYPE] = 'not support type %s '
MSG_CODE[M.CODE_SYSTEM_ERROR] = 'system error'

function M.get_msg_by_code(msg_code, params)
    local result = {}
    if MSG_CODE[msg_code] ~= nil then
        if not (params or false) then
            result = { code = msg_code, msg = MSG_CODE[msg_code] }
        else
            result = { code = msg_code, msg = string.format(MSG_CODE[msg_code], params) }
        end
    else
        result = { code = msg_code, msg = 'unknow' }
    end
    return result
end

return M

