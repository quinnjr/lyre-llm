import json
import os
import urllib.request

from lyre.instruments import Instrument, LABEL_VOCAB, program_for, normalize_label


def llm_from_env():
    endpoint = os.environ.get("LYRE_LLM_ENDPOINT")
    if not endpoint:
        return None
    return LLMLabeler(
        endpoint=endpoint,
        api_key=os.environ.get("LYRE_LLM_KEY", ""),
        model=os.environ.get("LYRE_LLM_MODEL", "gpt-4o-mini"),
    )


class LLMLabeler:
    def __init__(self, endpoint, api_key="", model="gpt-4o-mini", timeout=30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _prompt(self, instruments):
        rows = []
        for i, inst in enumerate(instruments):
            f = inst.features()
            rows.append(
                f"{i}: source={inst.source} name={inst.name!r} "
                f"notes={f['n_notes']} mean_polyphony={f['mean_polyphony']} "
                f"chordal_fraction={f['chordal_fraction']} single_line_fraction={f['single_line_fraction']} "
                f"pitch_range={f['pitch_min']}-{f['pitch_max']} duration_s={f['duration']}"
            )
        system = (
            "You label the instrument tracks of a multitrack transcription. "
            "Respond with JSON only: {\"tracks\":[{\"id\":<int>,\"label\":<str>}, ...]}. "
            f"Allowed labels: {LABEL_VOCAB}. A guitar source with both chordal and "
            "single-line passages should be split: emit two entries with the same id "
            "and labels 'guitar rhythm' and 'guitar lead'."
        )
        user = "\n".join(rows)
        return system, user

    def rename(self, instruments):
        system, user = self._prompt(instruments)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        content = payload["choices"][0]["message"]["content"]
        result = json.loads(content)
        out = []
        for entry in result.get("tracks", []):
            i = entry["id"]
            if i < 0 or i >= len(instruments):
                continue
            label = normalize_label(entry.get("label", instruments[i].name))
            base = instruments[i]
            out.append(
                Instrument(
                    name=label,
                    notes=base.notes,
                    program=program_for(label),
                    source=base.source,
                )
            )
        if not out:
            raise ValueError("LLM returned no tracks")
        return out
