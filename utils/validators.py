class Validators:
    @staticmethod
    def validate_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        return 'dienmayxanh.com' in url or 'tiki.vn' in url
    
    @staticmethod
    def validate_text(text: str) -> bool:
        return isinstance(text, str) and text.strip()