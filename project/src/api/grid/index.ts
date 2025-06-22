// 统一管理项目用户相关的接口
import request from "@/utils/request";

// 统一管理接口
enum API {
    GRID_URL = "api/v1/data/get_data"
}

export interface GridRow {
    longitude: number,
    latitude: number,
    value: number
}

export interface GridResponse {
    data: GridRow[]
}

// 暴露请求函数
// 登录接口方法
export const getGrid = (params: {data: string}): Promise<GridResponse> => {
  return request.get<GridResponse>(API.GRID_URL, { params }) as unknown as Promise<GridResponse>
};
