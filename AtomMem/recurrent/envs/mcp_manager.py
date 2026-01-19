#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCPManager客户端

此客户端用于与MCPManager的HTTP API进行交互
"""

import os
import json
import logging
import asyncio
import httpx
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("mcp_manager")


class MCPManager:
    """
    MCPManager客户端类
    用于与MCPManager的HTTP API进行交互，包括查询服务器、工具和执行工具调用
    """

    def __init__(self, manager_url: str = "http://localhost:8000/mcpapi", timeout: int = 125):
        """
        初始化MCPManager客户端

        Args:
            manager_url: MCPManager API的基础URL，默认为 http://localhost:8000/mcpapi
            timeout: 请求超时时间（秒）
        """
        self.manager_url = manager_url
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.servers = {}
        self.tools_by_server = {}
        self.all_tools = []
        
        # 兼容原MCPHandler的接口
        self.openai_tools = []

    async def initialize(self) -> bool:
        """
        初始化客户端

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 获取可用服务器列表
            logger.info(f"正在从 {self.manager_url} 初始化MCPManager客户端...")
            servers_response = await self.http_client.get(f"{self.manager_url}/servers")
            servers_response.raise_for_status()
            
            servers_data = servers_response.json()
            self.servers = {server: {} for server in servers_data.get("servers", [])}
            logger.info(f"获取到 {len(self.servers)} 个MCP服务器")

            # 获取所有服务器上的工具
            tools_response = await self.http_client.get(f"{self.manager_url}/tools")
            tools_response.raise_for_status()
            
            tools_data = tools_response.json()
            self.all_tools = tools_data.get("tools", [])
            logger.info(f"获取到 {len(self.all_tools)} 个MCP工具")
            
            # 按服务器组织工具
            for server in self.servers.keys():
                try:
                    server_tools_response = await self.http_client.get(f"{self.manager_url}/server/{server}/tools")
                    server_tools_response.raise_for_status()
                    
                    server_tools_data = server_tools_response.json()
                    self.tools_by_server[server] = server_tools_data.get("tools", [])
                    logger.info(f"服务器 '{server}' 上有 {len(self.tools_by_server[server])} 个工具")
                    
                    # 转换为OpenAI格式工具，以兼容原有代码
                    for tool in server_tools_data.get("tools", []):
                        if "function" in tool:
                            tool_function = tool["function"]
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool_function.get("name"),
                                    "description": tool_function.get("description", "").replace("(at most 3) URLs", "(at most 1) URLs"),
                                    "parameters": tool_function.get("parameters", {})
                                }
                            }
                            if tool_function.get("name") in ["search", "fetch_url"]:
                                self.openai_tools.append(openai_tool)
                    
                except Exception as e:
                    logger.error(f"获取服务器 '{server}' 的工具时出错: {str(e)}")
                    self.tools_by_server[server] = []

            logger.info(f"转换了 {len(self.openai_tools)} 个工具为OpenAI格式")
            return True
        except Exception as e:
            logger.error(f"初始化MCPManager客户端时出错: {str(e)}")
            return False
        
    async def list_servers(self) -> List[str]:
        """
        列出所有可用的MCP服务器

        Returns:
            List[str]: 服务器ID列表
        """
        return list(self.servers.keys())
    
    async def get_server_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """
        获取特定服务器上的工具

        Args:
            server_id: 服务器ID

        Returns:
            List[Dict[str, Any]]: 工具schema列表
        """
        if server_id not in self.tools_by_server:
            logger.warning(f"未知服务器ID: {server_id}")
            return []
        
        return self.tools_by_server[server_id]
    
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有服务器上的所有工具

        Returns:
            List[Dict[str, Any]]: 工具schema列表
        """
        return self.all_tools
    
    @retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数退避
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),  # 只对特定异常重试
    reraise=True  # 重试结束后仍然抛出异常
    )
    async def call_tool(self, 
                       tool_id: str, 
                       arguments: Dict[str, Any], 
                       server_id: Optional[str] = None) -> Dict[str, Any]:
        """
        调用MCP工具

        Args:
            tool_id: 工具ID
            arguments: 工具参数
            server_id: （可选）服务器ID，如果不提供则尝试自动查找

        Returns:
            Dict[str, Any]: 工具响应，总是包含status字段("success"或"error")
        """
        self.http_client = httpx.AsyncClient(timeout=self.timeout)
        try:
            # 如果提供了服务器ID和工具ID，构造完整的工具调用ID
            if server_id is not None and "." not in tool_id:
                full_tool_id = f"{server_id}.{tool_id}"
            else:
                full_tool_id = tool_id
            
            # 调用工具
            logger.info(f"调用工具: {full_tool_id}，参数: {arguments}")
            
            try:
                response = await self.http_client.post(
                    f"{self.manager_url}/tool/{full_tool_id}",
                    json=arguments,
                    timeout=self.timeout
                )
                
                # 检查HTTP状态码
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                logger.debug(f"工具调用返回结果: {result}")
                
                # 检查响应中是否包含错误信息
                if isinstance(result, dict):
                    # 如果已经有status字段，检查是否为错误
                    if "status" in result and result["status"] == "error":
                        error_msg = result.get("content", {}).get("error", "未知错误")
                        error_detail = result.get("content", {}).get("detail", "")
                        logger.error(f"工具 {full_tool_id} 返回错误状态: {error_msg}")
                        if error_detail:
                            logger.error(f"错误详情: {error_detail}")
                    # 如果content中包含error字段，设置status为error
                    elif isinstance(result.get("content"), dict) and "error" in result["content"]:
                        error_msg = result["content"]["error"]
                        logger.error(f"工具 {full_tool_id} 的响应content中包含错误: {error_msg}")
                        # 添加status字段
                        if "status" not in result:
                            result["status"] = "error"
                    # 如果没有错误信息，确保有status字段
                    elif "status" not in result:
                        result["status"] = "success"
                
                return result
                
            except httpx.ReadTimeout:
                logger.error(f"调用工具 {full_tool_id} 超时")
                return {
                    "status": "error",
                    "content": {
                        "error": "工具调用超时",
                        "detail": f"调用 {full_tool_id} 超时，请稍后重试",
                        "tool_id": full_tool_id
                    }
                }
            except httpx.HTTPStatusError as e:
                logger.error(f"调用工具 {full_tool_id} HTTP错误: {e.response.status_code} - {e.response.text}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"HTTP错误 {e.response.status_code}",
                        "detail": e.response.text,
                        "tool_id": full_tool_id
                    }
                }
            except httpx.RequestError as e:
                logger.error(f"调用工具 {full_tool_id} 请求错误: {str(e)}")
                return {
                    "status": "error",
                    "content": {
                        "error": f"请求错误",
                        "detail": str(e),
                        "tool_id": full_tool_id
                    }
                }
            
        except Exception as e:
            logger.error(f"调用工具 '{tool_id}' 时出错: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"错误详情: {tb}")
            
            return {
                "status": "error",
                "content": {
                    "error": str(e),
                    "detail": "调用工具时发生内部错误",
                    "traceback": tb,
                    "tool_id": tool_id,
                    "server_id": server_id
                }
            }