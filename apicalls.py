from dataclasses import dataclass
from typing import TypeVar
import requests
import json


ApiResponseInstance = TypeVar('ApiResponseInstance', bound='ApiResponse')

URL = "http://127.0.0.1:8000"


@dataclass
class ApiResponse:
    status_code: int
    headers: dict
    content: dict
    url: str

    @classmethod
    def from_requests_response(cls, response: requests.Response) -> ApiResponseInstance:
        return cls(response.status_code, dict(response.headers), json.loads(response.content), response.url)


class InvokeRestMethod:

    header = {"Content-Type": "application/json"}

    @staticmethod
    async def get(url, query_params=None) -> ApiResponseInstance:
        response = requests.get(url, params=query_params)
        return ApiResponse.from_requests_response(response)

    @classmethod
    async def post(cls, url, body: dict):
        response = requests.post(url, headers=cls.header, json=body)
        return ApiResponse.from_requests_response(response)

    @staticmethod
    def write_responses_to_file(responses: [ApiResponseInstance], file_name="apireturns.txt"):
        with open(file_name, "w") as file:
            for response in responses:
                file.write(f"url: {response.url}\n status_code: {response.status_code}\n "
                           f"headers: {response.headers}\n content: {response.content}\n\n")
        return True
