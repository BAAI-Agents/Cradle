from urllib.parse import quote
import datetime
import hashlib
import hmac
import http.client
import json
from typing import List, Dict

import dataclasses

REGION = 'us-west-2'
HOST = f'bedrock-runtime.{REGION}.amazonaws.com'
CONTENT_TYPE = 'application/json'
METHOD = 'POST'
SERVICE = 'bedrock'
SIGNED_HEADERS = 'host;x-amz-date'
CANONICAL_QUERY_STRING = ''
ALGORITHM = 'AWS4-HMAC-SHA256'


@dataclasses.dataclass
class RestfulClaudeClientResponse:
    id: str # The id of the message
    type: str # The type of the message
    role: str # The role of the message
    content: List[Dict[str, str]] # The content of the message
    model: str # The model used to generate the message
    stop_reason: str # The reason the model stopped generating the message
    stop_sequence: str # The sequence the model stopped generating the message
    usage: Dict[str, int] # The usage of the model


class RestfulClaudeClient():

    def __init__(self, llm_model, ak_val, sk_val):
        self.llm_model = llm_model
        self.ak_val = ak_val
        self.sk_val = sk_val

        self.model_id = f'anthropic.{self.llm_model}-v1:0'
        self.model_id = quote(self.model_id, safe='')
        self.bedrock_endpoint_url = f'https://{HOST}/model/{self.model_id}/invoke'
        self.bedrock_endpoint_stream_url = f'https://{HOST}/model/{self.model_id}/invoke-with-response-stream'


    def sign(self, key, msg):
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


    def get_canonical_url(self, endpoint_url):
        canonical_uri = '/' + '/'.join(endpoint_url.split('/')[-3:])
        return canonical_uri


    def get_date_time(self):
        date_time = datetime.datetime.utcnow()
        date = date_time.strftime('%Y%m%d')
        time = date_time.strftime('%Y%m%dT%H%M%SZ')
        return date, time


    def construct_signature_key(self, key, date_stamp, region_name, service_name):
        signature_date = self.sign(('AWS4' + key).encode('utf-8'), date_stamp)
        signature_region = self.sign(signature_date, region_name)
        signature_service = self.sign(signature_region, service_name)
        signature_key = self.sign(signature_service, 'aws4_request')
        return signature_key


    def construct_canonical_headers(self, time):
        canonical_headers = f'host:{HOST}\nx-amz-date:{time}\n'
        return canonical_headers


    def get_payload_hash(self, payload, type='str'):
        if type == 'str':
            payload = payload.encode('utf-8')
        else:
            payload = payload

        payload_hash = hashlib.sha256(payload).hexdigest()
        return payload_hash


    def construct_canonical_request(self, canonical_uri, canonical_headers, payload_hash):
        url_parts = canonical_uri.split('/')
        url_parts[2] = quote(url_parts[2], safe='')

        canonical_uri = '/'.join(url_parts)

        canonical_request = f'{METHOD}\n{canonical_uri}\n{CANONICAL_QUERY_STRING}\n{canonical_headers}\n{SIGNED_HEADERS}\n{payload_hash}'

        return canonical_request


    def get_credential_scope(self, date):
        credential_scope = f'{date}/{REGION}/{SERVICE}/aws4_request'
        return credential_scope


    def get_string_to_sign(self, time, credential_scope, canonical_request):
        digest = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = f'{ALGORITHM}\n{time}\n{credential_scope}\n{digest}'
        return string_to_sign


    def get_signing_key(self, date):
        key = self.construct_signature_key(self.sk_val, date, REGION, SERVICE)
        return key


    def get_signature(self, key, string_to_sign):
        signature = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        return signature


    def construct_authorization_headers(self, credential_scope, signature):
        authorization_headers = f'{ALGORITHM} Credential={self.ak_val}/{credential_scope}, SignedHeaders={SIGNED_HEADERS}, Signature={signature}'
        return authorization_headers


    def get_headers(self, time, authorization_headers):
        headers = {'X-Amz-Date': time,
                   'Authorization': authorization_headers}
        return headers


    def authorize(self, payload, payload_hash=None, stream=False):
        date, time = self.get_date_time()
        canonical_headers = self.construct_canonical_headers(time)

        if stream:
            canonical_uri = self.get_canonical_url(self.bedrock_endpoint_stream_url)
        else:
            canonical_uri = self.get_canonical_url(self.bedrock_endpoint_url)

        if payload_hash is None:
            payload_hash = self.get_payload_hash(payload)

        canonical_request = self.construct_canonical_request(canonical_uri, canonical_headers, payload_hash)

        credential_scope = self.get_credential_scope(date)
        string_to_sign = self.get_string_to_sign(time, credential_scope, canonical_request)

        signing_key = self.get_signing_key(date)
        signature = self.get_signature(signing_key, string_to_sign)

        authorization_headers = self.construct_authorization_headers(credential_scope, signature)

        headers = self.get_headers(time, authorization_headers)
        return headers


    def create(self,
               messages,
               system,
               temperature=1.0,
               max_tokens=1000,
               stream=False,
               ):

        conn = http.client.HTTPSConnection(HOST)

        PAYLOAD_0 = json.dumps({
            "system": system,
            "messages": messages,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "stop_sequences": ["\n\nHuman:", "\n\nAssistant"],
            "top_p": 0.999,
            "temperature": temperature,
        })

        payload_hash = self.get_payload_hash(PAYLOAD_0)

        headers = self.authorize(PAYLOAD_0, payload_hash, stream=stream)

        conn.request('POST', f'/model/{self.model_id}/invoke', PAYLOAD_0, headers)

        response = conn.getresponse()
        resp_body = json.loads(response.read().decode())

        response_data = RestfulClaudeClientResponse(
            id=resp_body['id'],
            type=resp_body['type'],
            role=resp_body['role'],
            content=resp_body['content'],
            model=resp_body['model'],
            stop_reason=resp_body['stop_reason'],
            stop_sequence=resp_body['stop_sequence'],
            usage=resp_body['usage'],
        )

        conn.close()
        return response_data
