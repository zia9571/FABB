import os
import warnings

try:
    import google.generativeai as genai
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception:
        pass
except Exception:
    genai = None


def generate_report(numeric_trace, citations, user_query):
    prompt_lines = ["You are a financial analysis assistant.", "", "Numeric Data:"]
    for key, value in numeric_trace.items():
        prompt_lines.append(f"- {key}: {value}")

    prompt_lines.append("")
    prompt_lines.append("Citations:")
    for i, cit in enumerate(citations, start=1):
        prompt_lines.append(f"{i}. {cit}")

    prompt_lines.append("")
    prompt_lines.append("User Question:")
    prompt_lines.append(user_query)
    prompt_lines.append("")
    prompt_lines.append("Provide a clear, concise answer based on the numeric data and citations. Include explanations.")
    prompt = "\n".join(prompt_lines)

    if genai is not None:
        try:
            if hasattr(genai, "generate_text"):
                resp = genai.generate_text(
                    model="gemini-pro-1",
                    prompt=prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                )
                if hasattr(resp, "result"):
                    return resp.result
                if isinstance(resp, dict) and "candidates" in resp:
                    return resp["candidates"][0].get("content", str(resp))
                return str(resp)

            if hasattr(genai, "Completion") and hasattr(genai.Completion, "create"):
                resp = genai.Completion.create(
                    model="gemini-pro-1",
                    prompt=prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                )
                if hasattr(resp, "text"):
                    return resp.text
                if isinstance(resp, dict):
                    return resp.get("text") or str(resp)
                return str(resp)

            if hasattr(genai, "chat") or hasattr(genai, "Chat"):
                try:
                    chat_mod = getattr(genai, "chat", getattr(genai, "Chat", None))
                    if chat_mod is not None:
                        out = chat_mod(model="gemini-pro-1").complete(prompt)
                        return str(out)
                except Exception:
                    pass

        except Exception as e:
            warnings.warn(f"GenAI call failed: {e}. Falling back to template response.")

    try:
        from_value = numeric_trace.get("from_value")
        to_value = numeric_trace.get("to_value")
        pct = numeric_trace.get("pct_change")
    except Exception:
        from_value = to_value = pct = None

    lines = []
    lines.append("[FALLBACK REPORT]")
    lines.append("")
    lines.append("Summary:")
    if from_value is not None and to_value is not None and pct is not None:
        lines.append(f"- From value: {from_value}")
        lines.append(f"- To value:   {to_value}")
        lines.append(f"- Change:     {pct}")
        # small interpretation
        try:
            pct_num = float(pct)
            if pct_num > 0:
                lines.append("Conclusion: The metric increased.")
            elif pct_num < 0:
                lines.append("Conclusion: The metric decreased.")
            else:
                lines.append("Conclusion: No change detected.")
        except Exception:
            pass
    else:
        lines.append("Could not compute numeric summary from inputs.")

    lines.append("")
    lines.append("Citations:")
    for cit in citations:
        lines.append(f"- {cit}")

    return "\n".join(lines)
