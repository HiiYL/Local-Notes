from __future__ import annotations

from typing import Iterable, Tuple, Optional, List, Dict, Any

import json as _json
import re
import json5
 

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from ..service import search_index

def _build_llm_cfg(model_tag: Optional[str]) -> dict:
    """Configure Qwen-Agent to use local Ollama (OpenAI-compatible) with qwen3:8b by default."""
    return {
        'model': model_tag or 'qwen3:8b',
        'model_server': 'http://localhost:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.8,
        },
    }

@register_tool('search_notes')
class SearchNotesTool(BaseTool):
    description = 'Search your local notes index for relevant snippets and return ranked items.'
    parameters = [
        { 'name': 'query', 'type': 'string', 'description': 'Query to search in your notes', 'required': True },
        { 'name': 'k', 'type': 'integer', 'description': 'Number of results (default 6)', 'required': False },
    ]

    def call(self, params: str, **kwargs) -> str:
        try:
            args = json5.loads(params) if isinstance(params, str) else (params or {})
        except Exception:
            args = {}
        query = (args.get('query') or '').strip()
        k = int(args.get('k') or 6)
        # Query vectorstore via service helper (uses defaults if not passed)
        try:
            results = search_index(query=query, k=k, max_chars=300)
        except Exception as e:
            print('search_notes tool error:', repr(e))
            results = []
        out: List[Dict[str, Any]] = []
        for r in (results or []):
            out.append({
                'rank': r.get('rank'),
                'title': r.get('title', ''),
                'folder': r.get('folder', ''),
                'chunk': r.get('chunk', 0),
                'text': r.get('text', ''),
                'heading': r.get('heading', ''),
                'updated_at': r.get('updated_at', 0),
            })
        return _json.dumps(out, ensure_ascii=False)

def stream_qwen_agent(
    question: str,
    provider: str = "ollama",
    llm_model: Optional[str] = None,
    max_tool_iters: int = 2,
) -> Iterable[Tuple[str, str]]:
    """Qwen-Agent with a registered `search_notes` tool (RAG-only). Streams tool/sources/delta.

    We instruct the assistant to first call `search_notes` with the user query, then answer citing [n].
    """
    # Module-level event queue for tool + sources emissions from the tool class
    events: List[Tuple[str, str]] = []

    def emit(ev: str, data: str):
        events.append((ev, data))

    # Tool implementation scoped to this request so it can emit events

    # Build assistant
    llm_cfg = _build_llm_cfg(llm_model)
    system_message = (
        "You are a helpful assistant. Use tools to fetch evidence from notes before answering.\n"
        "First, call the `search_notes` tool with the user query to retrieve snippets.\n"
        "Then answer concisely using only those snippets, citing ranks like [1], [2].\n\n"
        "OUTPUT RULES (strict):\n"
        "- Respond in Markdown.\n"
        "- Start with a bolded line: **Top results** (or an answer heading).\n"
        "- Follow with a numbered list, one item per line. Each item must be on its own line, not inline.\n"
        "- Each item: Title — short summary [rank].\n"
        "- Use blank lines between paragraphs (two newlines).\n"
        "- End with a 'Sources' line listing the cited numbers in order, e.g., Sources: [1], [2], [5].\n"
        "- Do not merge multiple numbered items into one paragraph. Keep them as separate list lines."
    )
    bot = Assistant(llm=llm_cfg, system_message=system_message, function_list=['search_notes'])

    # Build chat messages
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": question},
    ]

    # Helper to robustly extract display text from various Qwen-Agent streaming payloads
    def _extract_text(resp: Any) -> str:
        # print("EXTRACTING TEXT", resp)
        # Plain string
        if isinstance(resp, str):
            return resp
        # Dict payloads
        if isinstance(resp, dict):
            # Common: {'role':'assistant','content': '...'}
            c = resp.get('content')
            if isinstance(c, str):
                return c
            # {'content': [{'text': '...'} or {'type':'text','text':'...'} ...]}
            if isinstance(c, list):
                parts: List[str] = []
                for it in c:
                    if isinstance(it, str):
                        parts.append(it)
                    elif isinstance(it, dict):
                        if isinstance(it.get('text'), str):
                            parts.append(it['text'])
                        elif isinstance(it.get('delta'), str):
                            parts.append(it['delta'])
                return ''.join(parts)
            # Some runners use 'delta' or 'response' keys directly
            for key in ('delta', 'response', 'text'):
                v = resp.get(key)
                if isinstance(v, str):
                    return v
        # List of message dicts
        if isinstance(resp, list):
            parts: List[str] = []
            for m in resp:
                if isinstance(m, dict) and m.get('role') == 'assistant':
                    mc = m.get('content')
                    if isinstance(mc, str):
                        parts.append(mc)
                    elif isinstance(mc, list):
                        for it in mc:
                            if isinstance(it, str):
                                parts.append(it)
                            elif isinstance(it, dict) and isinstance(it.get('text'), str):
                                parts.append(it['text'])
            return ''.join(parts)
        return ''

    # Helpers to surface tool and sources events from streamed messages
    def _handle_tools_and_sources(resp: Any) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        def handle_msg(m: Dict[str, Any]):
            # assistant function_call => tool event
            if m.get('role') == 'assistant' and isinstance(m.get('function_call'), dict):
                fc = m['function_call']
                name = fc.get('name')
                args = fc.get('arguments')
                if name:
                    # Safely parse arguments which may be an empty string, a JSON string, or a dict
                    parsed_args: Dict[str, Any] = {}
                    try:
                        if isinstance(args, str):
                            if args.strip():
                                parsed_args = json5.loads(args)
                            else:
                                parsed_args = {}
                        elif isinstance(args, dict):
                            parsed_args = args
                        else:
                            parsed_args = {}
                    except Exception:
                        parsed_args = {}
                    payload = _json.dumps({'name': name, 'args': parsed_args})
                    events.append(('tool', payload))
            # function role with tool result => sources (for search_notes)
            if m.get('role') == 'function':
                fname = m.get('name')
                content = m.get('content')
                if fname == 'search_notes' and isinstance(content, str) and content.strip():
                    # content is JSON string returned by tool
                    events.append(('sources', content))

        if isinstance(resp, list):
            for m in resp:
                if isinstance(m, dict):
                    handle_msg(m)
        elif isinstance(resp, dict):
            handle_msg(resp)
        return events

    # Run and stream
    try:
        # Maintain cumulative assistant content and parse newly added spans
        cumulative: str = ''
        in_think: bool = False
        current_think_id: Optional[int] = None
        next_think_id: int = 0
        tag_re = re.compile(r"(<\/?think>)", re.IGNORECASE)
        # Track sources and citations
        latest_sources: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        emitted_citations: set[int] = set()

        tool_seen: set[str] = set()
        for resp in bot.run(messages=messages):
            # Derive tool and sources events from the stream itself
            for e in _handle_tools_and_sources(resp):
                etype, edata = e
                if etype == 'tool':
                    # Deduplicate exact same tool payloads within a run
                    key = edata
                    if key in tool_seen:
                        continue
                    tool_seen.add(key)
                    yield (etype, edata)
                elif etype == 'sources':
                    # Cache parsed sources for later citation filtering, but DO NOT emit yet
                    try:
                        latest_sources = json5.loads(edata) or []
                    except Exception:
                        latest_sources = []
                    # no yield here; only show sources when cited
                else:
                    yield (etype, edata)

            current = _extract_text(resp) or ''
            if not current:
                yield ('delta', '\u200b')
                continue

            # Compute newly added portion relative to cumulative
            added = current[len(cumulative):] if current.startswith(cumulative) else current
            if len(current) >= len(cumulative):
                cumulative = current
            if not added:
                continue

            # Parse the added chunk for think tags and emit timeline events
            parts = tag_re.split(added)
            for part in parts:
                if part.lower() == '<think>':
                    if not in_think:
                        next_think_id += 1
                        current_think_id = next_think_id
                        in_think = True
                        yield ('thinking_start', str(current_think_id))
                    # else nested think, ignore start
                elif part.lower() == '</think>':
                    if in_think and current_think_id is not None:
                        yield ('thinking_end', str(current_think_id))
                    in_think = False
                    current_think_id = None
                else:
                    if not part:
                        continue
                    if in_think and current_think_id is not None:
                        payload = _json.dumps({'id': current_think_id, 'text': part})
                        yield ('thinking', payload)
                    else:
                        # visible answer text
                        full_text_parts.append(part)
                        yield ('delta', part)
                        # compute citations incrementally
                        try:
                            partial = ''.join(full_text_parts)
                            cited_now = set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", partial))
                            if latest_sources and (cited_now - emitted_citations):
                                emitted_citations = cited_now
                                cited_list = [s for s in latest_sources if int(s.get('rank', 0)) in cited_now]
                                yield ('citations', _json.dumps(cited_list))
                        except Exception:
                            pass
    except Exception as e:
        yield ('delta', f'(agent error) {e}')
    # After loop completes, emit final citations set if any
    try:
        final_text = ''.join(full_text_parts)
        cited = set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", final_text))
        if latest_sources and cited:
            cited_list = [s for s in latest_sources if int(s.get('rank', 0)) in cited]
            yield ('citations', _json.dumps(cited_list))
    except Exception:
        pass
