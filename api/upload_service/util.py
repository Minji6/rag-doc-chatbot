def map_codes(value: str, code_map: dict[str, str]) -> str:
    if not value:
        return ""

    return ",".join(
        code_map.get(code.strip(), code.strip())
        for code in value.split(",")
        if code.strip()
    )