import fitz
import random

class TamperEngine:
    
    @staticmethod
    def apply_attack(original_path):
        try:
            doc = fitz.open(original_path)
            return TamperEngine.modify_document(doc)
        except Exception:
            return TamperEngine.create_dummy_pdf()
    
    @staticmethod
    def modify_document(doc):
        if doc.page_count == 0:
            return doc
        
        page = doc[0]
        attack_type = random.choice(['text', 'rect', 'line', 'circle'])
        
        if attack_type == 'text':
            x = random.randint(50, 300)
            y = random.randint(50, 500)
            page.insert_text((x, y), f"TAMPERED_{random.randint(1000, 9999)}", color=(1, 0, 0))
        
        elif attack_type == 'rect':
            x1 = random.randint(50, 200)
            y1 = random.randint(50, 200)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(50, 100)
            page.draw_rect(fitz.Rect(x1, y1, x2, y2), color=(0, 1, 0), width=2)
        
        elif attack_type == 'line':
            x1 = random.randint(50, 200)
            y1 = random.randint(50, 200)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(50, 100)
            page.draw_line((x1, y1), (x2, y2), color=(0, 0, 1), width=2)
        
        elif attack_type == 'circle':
            x = random.randint(100, 300)
            y = random.randint(100, 400)
            r = random.randint(20, 50)
            page.draw_circle((x, y), r, color=(1, 0, 1), width=2)
        
        return doc
    
    @staticmethod
    def create_dummy_pdf():
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), "Synthetic Training Document")
        page.insert_text((50, 100), f"ID: {random.randint(10000, 99999)}")
        return doc