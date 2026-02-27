import requests
import json
import base64
import io
import time
import re
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class OpenAICompatibleV2Node:
    """
    ComfyUI node for broader OpenAI-compatible API calls.
    Keeps v1 node behavior untouched and adds extra compatibility controls.
    """

    API_MODES = ["auto", "chat_completions", "responses"]
    TOKEN_FIELDS = ["auto", "max_tokens", "max_completion_tokens", "max_output_tokens", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "model_name": ("STRING", {
                    "default": "gpt-4o",
                }),
                "api_url": ("STRING", {
                    "default": "https://api.openai.com/v1/chat/completions",
                }),
                "api_key": ("STRING", {
                    "default": "",
                }),
                "max_tokens": ("INT", {
                    "default": 4096,
                    "min": 1,
                    "max": 128000,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                }),
                "use_seed": ("BOOLEAN", {
                    "default": False,
                }),
                "api_mode": (cls.API_MODES, {
                    "default": "auto",
                }),
                "token_field": (cls.TOKEN_FIELDS, {
                    "default": "auto",
                }),
                "auto_retry_on_fail": ("BOOLEAN", {
                    "default": True,
                }),
                "max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                }),
                "retry_backoff": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                }),
                "timeout_seconds": ("INT", {
                    "default": 120,
                    "min": 5,
                    "max": 1200,
                    "step": 5,
                }),
                "download_max_retries": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                }),
                "download_retry_backoff": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                }),
                "response_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "extra_headers_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                }),
                "extra_body_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                }),
                "download_markdown_image": ("BOOLEAN", {
                    "default": True,
                }),
                "strip_think_tag": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "IMAGE"
    )
    RETURN_NAMES = (
        "response", "final_answer", "status", "http_status", "model_name", "api_url", "api_key", "max_tokens", "user_prompt", "raw_json", "image"
    )
    FUNCTION = "call_api"
    CATEGORY = "scvxzf/AI/Text"

    def tensor_to_base64(self, tensor):
        if tensor is None:
            return None

        img_tensor = tensor[0] if len(tensor.shape) == 4 else tensor
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return base64_str

    def strip_think(self, text):
        if not isinstance(text, str):
            return str(text)
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text

    def parse_json_object(self, value, field_name):
        if value is None:
            return {}
        value = str(value).strip()
        if not value:
            return {}
        try:
            data = json.loads(value)
        except Exception as e:
            raise ValueError(f"Invalid JSON in {field_name}: {str(e)}")

        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a JSON object.")
        return data

    def merge_dict(self, base_obj, override_obj):
        if not isinstance(base_obj, dict):
            return override_obj
        if not isinstance(override_obj, dict):
            return base_obj

        merged = dict(base_obj)
        for key, value in override_obj.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_dict(merged[key], value)
            else:
                merged[key] = value
        return merged

    def detect_mode(self, api_mode, api_url):
        if api_mode != "auto":
            return api_mode

        lower_url = (api_url or "").lower()
        if "/responses" in lower_url:
            return "responses"
        return "chat_completions"

    def select_token_field(self, token_field, mode):
        if token_field != "auto":
            return token_field
        if mode == "responses":
            return "max_output_tokens"
        return "max_tokens"

    def create_chat_messages(self, system_prompt, user_prompt, image):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        if image is not None:
            base64_image = self.tensor_to_base64(image)
            user_content = []
            if user_prompt and user_prompt.strip():
                user_content.append({"type": "text", "text": user_prompt})
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        return messages

    def create_responses_input(self, system_prompt, user_prompt, image):
        items = []
        if system_prompt and system_prompt.strip():
            items.append({"role": "system", "content": system_prompt})

        if image is not None:
            base64_image = self.tensor_to_base64(image)
            content_items = []
            if user_prompt and user_prompt.strip():
                content_items.append({"type": "input_text", "text": user_prompt})
            content_items.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}"
            })
            items.append({"role": "user", "content": content_items})
        else:
            items.append({"role": "user", "content": user_prompt})

        return items

    def try_path(self, data, path):
        if not path or not isinstance(path, str):
            return None, False

        current = data
        parts = [p for p in path.split(".") if p != ""]

        for part in parts:
            if isinstance(current, list):
                if not part.isdigit():
                    return None, False
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None, False
                current = current[idx]
            elif isinstance(current, dict):
                if part not in current:
                    return None, False
                current = current[part]
            else:
                return None, False

        return current, True

    def to_text(self, value, depth=0):
        if value is None:
            return ""

        if depth > 8:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, (int, float, bool)):
            return str(value)

        if isinstance(value, list):
            parts = []
            for item in value:
                text = self.to_text(item, depth + 1)
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()

        if isinstance(value, dict):
            # Common text keys used by OpenAI-compatible vendors.
            for key in ["output_text", "text", "content", "response", "answer", "result"]:
                if key in value:
                    text = self.to_text(value.get(key), depth + 1)
                    if text:
                        return text

            # Chat-completions style nested message.
            if "message" in value:
                text = self.to_text(value.get("message"), depth + 1)
                if text:
                    return text

            # Generic fallback for dict objects with type markers.
            if "type" in value and "text" in value:
                text = self.to_text(value.get("text"), depth + 1)
                if text:
                    return text

            return ""

        return ""

    def extract_response_text(self, result, response_path):
        # 1) User-defined path has top priority for vendor-specific schemas.
        value, ok = self.try_path(result, response_path)
        if ok:
            text = self.to_text(value)
            if text:
                return text

        # 2) OpenAI chat-completions style.
        choices = result.get("choices") if isinstance(result, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0]
            text = self.to_text(first)
            if text:
                return text

        # 3) OpenAI responses style: output_text / output[].content[].
        if isinstance(result, dict):
            text = self.to_text(result.get("output_text"))
            if text:
                return text

            output = result.get("output")
            text = self.to_text(output)
            if text:
                return text

            # 4) Common vendor keys.
            for key in ["response", "answer", "result", "content", "text", "message"]:
                text = self.to_text(result.get(key))
                if text:
                    return text

        # 5) Fallback to compact JSON string.
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    def build_payload(self, mode, system_prompt, user_prompt, temperature, model_name, max_tokens, token_field, image, extra_body):
        if mode == "responses":
            payload = {
                "model": model_name,
                "input": self.create_responses_input(system_prompt, user_prompt, image),
                "temperature": temperature,
            }
        else:
            payload = {
                "model": model_name,
                "messages": self.create_chat_messages(system_prompt, user_prompt, image),
                "temperature": temperature,
            }

        selected_token_field = self.select_token_field(token_field, mode)
        if selected_token_field != "none":
            payload[selected_token_field] = max_tokens

        payload = self.merge_dict(payload, extra_body)
        return payload

    def resolve_seed(self, seed):
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = -1

        if seed_value < 0:
            seed_value = random.randint(0, 2147483647)
        return seed_value

    def should_retry(self, status_code):
        if status_code == 429:
            return True
        if status_code >= 500:
            return True
        return False

    def normalize_bool(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ["1", "true", "yes", "on"]
        return False

    def wants_stream(self, payload):
        if not isinstance(payload, dict):
            return False
        return self.normalize_bool(payload.get("stream", False))

    def parse_stream_event(self, event_lines):
        if not event_lines:
            return None, False, None

        data_lines = []
        raw_lines = []

        for line in event_lines:
            if not line:
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                continue
            lower = line.lower()
            if lower.startswith("event:") or lower.startswith("id:") or lower.startswith("retry:"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            else:
                raw_lines.append(line)

        if data_lines:
            data_text = "\n".join(data_lines).strip()
        else:
            data_text = "\n".join(raw_lines).strip()

        if not data_text:
            return None, False, None

        if data_text == "[DONE]":
            return None, True, None

        try:
            return json.loads(data_text), False, None
        except json.JSONDecodeError as e:
            return None, False, f"Invalid stream JSON chunk: {str(e)}"

    def extract_stream_chunk_text(self, chunk):
        if chunk is None:
            return ""

        if not isinstance(chunk, dict):
            return self.to_text(chunk)

        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict):
                    for key in ["content", "text", "reasoning_content"]:
                        text = self.to_text(delta.get(key))
                        if text:
                            return text
                elif isinstance(delta, str):
                    if delta:
                        return delta

                # Some providers still put full message in streaming chunks.
                text = self.to_text(first.get("message"))
                if text:
                    return text

        # Responses/event-stream style fallbacks.
        for key in ["output_text", "delta", "text", "content", "response", "answer", "result"]:
            text = self.to_text(chunk.get(key))
            if text:
                return text

        text = self.to_text(chunk.get("output"))
        if text:
            return text

        return ""

    def consume_stream_response(self, response):
        parts = []
        usage = {}
        finish_reason = ""
        done = False
        chunk_count = 0
        parse_errors = []
        event_lines = []

        def handle_event(lines):
            nonlocal usage, finish_reason, done, chunk_count

            chunk, is_done, err = self.parse_stream_event(lines)
            if err:
                parse_errors.append(err)
                return

            if is_done:
                done = True
                return

            if chunk is None:
                return

            chunk_count += 1
            if isinstance(chunk, dict):
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk.get("usage")

                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        reason = first.get("finish_reason")
                        if reason:
                            finish_reason = str(reason)

            text_piece = self.extract_stream_chunk_text(chunk)
            if text_piece:
                parts.append(text_piece)

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.rstrip("\r")
            stripped = line.strip()

            if not stripped:
                if event_lines:
                    handle_event(event_lines)
                    event_lines = []
                continue

            # Handle NDJSON/SSE hybrid where each JSON object is one line.
            if stripped == "[DONE]":
                if event_lines:
                    handle_event(event_lines)
                    event_lines = []
                done = True
                continue

            if stripped.startswith("{") and event_lines and not event_lines[-1].lstrip().startswith("data:"):
                handle_event(event_lines)
                event_lines = []

            event_lines.append(line)

        if event_lines:
            handle_event(event_lines)

        content = "".join(parts)
        stream_meta = {
            "mode": "stream",
            "chunks": chunk_count,
            "done": done,
            "finish_reason": finish_reason,
            "usage": usage if usage else None,
        }
        if parse_errors:
            stream_meta["parse_errors"] = parse_errors[:5]

        raw_json = json.dumps(stream_meta, ensure_ascii=False)

        if chunk_count == 0 and parse_errors:
            return "", raw_json, "Error: Stream parsing failed."

        return content, raw_json, None

    def extract_markdown_image_urls(self, text):
        if not isinstance(text, str) or not text.strip():
            return []

        # Markdown image syntax: ![alt](url)
        markdown_pattern = r"!\[[^\]]*\]\((https?://[^)\s]+)\)"
        urls = re.findall(markdown_pattern, text)

        # Fallback: direct image URLs in plain text.
        if not urls:
            direct_pattern = r"(https?://[^\s)]+(?:\.png|\.jpg|\.jpeg|\.webp|\.gif)(?:\?[^\s)]*)?)"
            urls = re.findall(direct_pattern, text, flags=re.IGNORECASE)

        unique_urls = []
        seen = set()
        for url in urls:
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def url_to_tensor(self, url, timeout_seconds, max_retries=0, retry_backoff=1.0):
        # Clamp download timeout so very long model timeout does not block download too long.
        download_timeout = max(5, min(int(timeout_seconds), 120))
        attempts = max(1, int(max_retries) + 1)
        last_error = "Unknown download error"

        for attempt in range(attempts):
            try:
                response = requests.get(url, timeout=download_timeout)
                response.raise_for_status()

                pil_image = Image.open(io.BytesIO(response.content))
                pil_image = pil_image.convert("RGB")

                img_np = np.array(pil_image).astype(np.float32) / 255.0
                return torch.from_numpy(img_np).unsqueeze(0)
            except Exception as e:
                last_error = str(e)
                if attempt < attempts - 1:
                    sleep_seconds = float(retry_backoff) * (2 ** attempt)
                    time.sleep(sleep_seconds)
                    continue
                break

        raise Exception(f"Failed after {attempts} attempts: {last_error}")

    def resolve_output_image(self, response_text, fallback_image, timeout_seconds, download_max_retries, download_retry_backoff):
        urls = self.extract_markdown_image_urls(response_text)
        if not urls:
            return fallback_image, "", [], ""

        # Prefer the latest generated result when multiple images are returned.
        selected_url = urls[-1]

        try:
            output_image = self.url_to_tensor(
                selected_url,
                timeout_seconds,
                max_retries=download_max_retries,
                retry_backoff=download_retry_backoff,
            )
            return output_image, "", urls, selected_url
        except Exception as e:
            return fallback_image, str(e), urls, selected_url

    def enrich_raw_json_with_image_meta(self, raw_json, urls, selected_url, image_error):
        if not urls and not image_error:
            return raw_json

        image_meta = {
            "detected_urls": urls,
            "selected_url": selected_url,
            "download_ok": image_error == "",
            "download_error": image_error if image_error else None,
        }

        if not raw_json:
            return json.dumps({"image_extract": image_meta}, ensure_ascii=False)

        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                parsed["image_extract"] = image_meta
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        wrapped = {
            "raw_text": raw_json,
            "image_extract": image_meta,
        }
        return json.dumps(wrapped, ensure_ascii=False)

    def make_return(self, response, final_answer, status, http_status, model_name, api_url, api_key, max_tokens, user_prompt, raw_json, image):
        return (
            response,
            final_answer,
            status,
            http_status,
            model_name,
            api_url,
            api_key,
            max_tokens,
            user_prompt,
            raw_json,
            image,
        )

    def call_api(self, system_prompt, user_prompt, temperature, model_name, api_url, api_key, max_tokens, seed, use_seed,
                 api_mode, token_field, auto_retry_on_fail, max_retries, retry_backoff, timeout_seconds, download_max_retries, download_retry_backoff,
                 response_path, extra_headers_json, extra_body_json, download_markdown_image, strip_think_tag, image=None):

        def error_result(msg, http_status=0, raw_json=""):
            final_answer = self.strip_think(msg) if strip_think_tag else msg
            return self.make_return(
                msg,
                final_answer,
                "error",
                http_status,
                model_name,
                api_url,
                api_key,
                max_tokens,
                user_prompt,
                raw_json,
                image,
            )

        if not api_key:
            return error_result("Error: API key is required.")

        if not (user_prompt and user_prompt.strip()) and image is None:
            return error_result("Error: User prompt or image is required.")

        try:
            extra_headers = self.parse_json_object(extra_headers_json, "extra_headers_json")
            extra_body = self.parse_json_object(extra_body_json, "extra_body_json")
        except ValueError as e:
            return error_result(f"Error: {str(e)}")

        mode = self.detect_mode(api_mode, api_url)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        headers = self.merge_dict(headers, extra_headers)

        payload = self.build_payload(
            mode=mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            model_name=model_name,
            max_tokens=max_tokens,
            token_field=token_field,
            image=image,
            extra_body=extra_body,
        )
        used_seed = None
        if use_seed:
            used_seed = self.resolve_seed(seed)
            payload["seed"] = used_seed

        raw_json = ""
        last_error = "Error: Unknown error."
        max_attempts = (max_retries + 1) if auto_retry_on_fail else 1

        for attempt in range(max_attempts):
            try:
                stream_requested = self.wants_stream(payload)

                with requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                    stream=stream_requested,
                ) as response:
                    http_status = response.status_code

                    if response.ok:
                        content_type = (response.headers.get("Content-Type") or "").lower()
                        is_stream_response = stream_requested or ("text/event-stream" in content_type)

                        if is_stream_response:
                            content, raw_json, stream_error = self.consume_stream_response(response)
                            if stream_error:
                                return error_result(stream_error, http_status=http_status, raw_json=raw_json)
                        else:
                            try:
                                result = response.json()
                                raw_json = json.dumps(result, ensure_ascii=False)
                            except json.JSONDecodeError:
                                result = {"raw_text": response.text}
                                raw_json = response.text

                            content = self.extract_response_text(result, response_path)

                        final_answer = self.strip_think(content) if strip_think_tag else content
                        output_image = image

                        if download_markdown_image:
                            output_image, image_error, found_urls, selected_url = self.resolve_output_image(
                                content,
                                image,
                                timeout_seconds,
                                download_max_retries,
                                download_retry_backoff,
                            )
                            raw_json = self.enrich_raw_json_with_image_meta(
                                raw_json,
                                found_urls,
                                selected_url,
                                image_error,
                            )
                        try:
                            parsed_raw = json.loads(raw_json) if raw_json else {}
                        except Exception:
                            parsed_raw = {"raw_text": raw_json}
                        if isinstance(parsed_raw, dict):
                            parsed_raw["used_seed"] = used_seed
                            parsed_raw["use_seed"] = bool(use_seed)
                            parsed_raw["auto_retry_on_fail"] = bool(auto_retry_on_fail)
                            parsed_raw["request_attempts"] = max_attempts
                            raw_json = json.dumps(parsed_raw, ensure_ascii=False)

                        return self.make_return(
                            content,
                            final_answer,
                            "success",
                            http_status,
                            model_name,
                            api_url,
                            api_key,
                            max_tokens,
                            user_prompt,
                            raw_json,
                            output_image,
                        )

                    # HTTP error path.
                    body_text = ""
                    try:
                        body_text = response.text[:800]
                    except Exception:
                        body_text = ""

                    last_error = f"Error: HTTP {http_status}. {body_text}".strip()

                    if attempt < max_attempts - 1 and self.should_retry(http_status):
                        sleep_seconds = retry_backoff * (2 ** attempt)
                        time.sleep(sleep_seconds)
                        continue

                    return error_result(last_error, http_status=http_status, raw_json=body_text)

            except requests.exceptions.Timeout:
                last_error = f"Error: Request timed out after {timeout_seconds}s."
            except requests.exceptions.RequestException as e:
                last_error = f"Error: {str(e)}"
            except Exception as e:
                last_error = f"Error: {str(e)}"

            if attempt < max_attempts - 1:
                sleep_seconds = retry_backoff * (2 ** attempt)
                time.sleep(sleep_seconds)
                continue

            return error_result(last_error)

        return error_result(last_error)


class OpenAICompatibleV2MultiImageNode(OpenAICompatibleV2Node):
    """
    Download and output all markdown images from one response as IMAGE batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        config = copy.deepcopy(super().INPUT_TYPES())
        config["required"].pop("download_markdown_image", None)
        config["required"]["max_images"] = ("INT", {
            "default": 8,
            "min": 1,
            "max": 64,
            "step": 1,
        })
        return config

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("response", "final_answer", "status", "http_status", "images", "image_count", "image_urls", "raw_json")
    FUNCTION = "call_api_multi"
    CATEGORY = "scvxzf/AI/Text"

    def create_placeholder_image(self, width=512, height=512):
        img_np = np.zeros((height, width, 3), dtype=np.float32)
        img_np[:, :, 0] = 0.2
        return torch.from_numpy(img_np).unsqueeze(0)

    def ensure_batch_tensor(self, tensor):
        if tensor is None:
            return None
        if len(tensor.shape) == 4:
            return tensor
        if len(tensor.shape) == 3:
            return tensor.unsqueeze(0)
        return None

    def resize_batch_to(self, tensor, target_h, target_w):
        if tensor.shape[1] == target_h and tensor.shape[2] == target_w:
            return tensor
        bchw = tensor.permute(0, 3, 1, 2)
        resized = F.interpolate(
            bchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1)

    def stack_image_batches(self, tensors):
        batches = []
        for tensor in tensors:
            batch = self.ensure_batch_tensor(tensor)
            if batch is not None:
                batches.append(batch)

        if not batches:
            return None

        target_h = batches[0].shape[1]
        target_w = batches[0].shape[2]
        normalized = []
        for batch in batches:
            normalized.append(self.resize_batch_to(batch, target_h, target_w))
        return torch.cat(normalized, dim=0)

    def enrich_raw_json_with_multi_image_meta(self, raw_json, urls, success_count, output_count, failed_errors):
        image_meta = {
            "detected_urls": urls,
            "detected_count": len(urls),
            "downloaded_count": success_count,
            "output_count": output_count,
            "failed_count": len(failed_errors),
            "failed_errors": failed_errors[:10],
        }

        if not raw_json:
            return json.dumps({"image_extract_multi": image_meta}, ensure_ascii=False)

        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                parsed["image_extract_multi"] = image_meta
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

        wrapped = {
            "raw_text": raw_json,
            "image_extract_multi": image_meta,
        }
        return json.dumps(wrapped, ensure_ascii=False)

    def call_api_multi(self, system_prompt, user_prompt, temperature, model_name, api_url, api_key, max_tokens, seed, use_seed,
                       api_mode, token_field, auto_retry_on_fail, max_retries, retry_backoff, timeout_seconds, download_max_retries, download_retry_backoff,
                       response_path, extra_headers_json, extra_body_json, max_images, strip_think_tag, image=None):

        base_result = super().call_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            model_name=model_name,
            api_url=api_url,
            api_key=api_key,
            max_tokens=max_tokens,
            seed=seed,
            use_seed=use_seed,
            api_mode=api_mode,
            token_field=token_field,
            auto_retry_on_fail=auto_retry_on_fail,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            timeout_seconds=timeout_seconds,
            download_max_retries=download_max_retries,
            download_retry_backoff=download_retry_backoff,
            response_path=response_path,
            extra_headers_json=extra_headers_json,
            extra_body_json=extra_body_json,
            download_markdown_image=False,
            strip_think_tag=strip_think_tag,
            image=image,
        )

        response = base_result[0]
        final_answer = base_result[1]
        status = base_result[2]
        http_status = base_result[3]
        raw_json = base_result[9]
        output_image = base_result[10]

        if status != "success":
            fallback_batch = self.ensure_batch_tensor(output_image)
            if fallback_batch is None:
                fallback_batch = self.create_placeholder_image()
            return (response, final_answer, status, http_status, fallback_batch, 0, "", raw_json)

        urls = self.extract_markdown_image_urls(response)
        urls = urls[:max_images]

        output_tensors = []
        failed_errors = []
        downloaded_count = 0

        # 批量下载响应里的所有图片链接，并统一成 ComfyUI IMAGE 批量输出
        for url in urls:
            try:
                output_tensors.append(
                    self.url_to_tensor(
                        url,
                        timeout_seconds,
                        max_retries=download_max_retries,
                        retry_backoff=download_retry_backoff,
                    )
                )
                downloaded_count += 1
            except Exception as e:
                failed_errors.append(f"{url} -> {str(e)}")
                # 下载失败时补占位图，保证多图数量与链接数量一致
                output_tensors.append(self.create_placeholder_image())

        image_batch = self.stack_image_batches(output_tensors)

        if image_batch is None:
            fallback_batch = self.ensure_batch_tensor(output_image)
            if fallback_batch is None:
                fallback_batch = self.create_placeholder_image()
            image_batch = fallback_batch

        image_count = int(image_batch.shape[0]) if len(image_batch.shape) == 4 else 0
        image_urls = "\n".join(urls)
        enriched_raw_json = self.enrich_raw_json_with_multi_image_meta(
            raw_json,
            urls,
            downloaded_count,
            image_count,
            failed_errors,
        )

        final_status = status
        if urls and downloaded_count == 0:
            final_status = "partial_success"
        elif downloaded_count < len(urls):
            final_status = "partial_success"

        return (
            response,
            final_answer,
            final_status,
            http_status,
            image_batch,
            image_count,
            image_urls,
            enriched_raw_json,
        )


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleV2": OpenAICompatibleV2Node,
    "OpenAICompatibleV2MultiImage": OpenAICompatibleV2MultiImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleV2": "OpenAI 兼容接口 2.0",
    "OpenAICompatibleV2MultiImage": "OpenAI 兼容接口 2.0 多图",
}
