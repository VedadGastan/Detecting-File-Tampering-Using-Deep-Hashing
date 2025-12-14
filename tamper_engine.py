import fitz
import random

class TamperEngine:
    
    @staticmethod
    def apply_attack(original_path):
        try:
            doc = fitz.open(original_path)
            attack_type = random.choice([
                'clean', 'clean', 'meta', 'annot', 'page', 
                'link', 'text', 'embed', 'layer', 'code'
            ])
            
            if attack_type == 'clean':
                return doc, 0.0
            
            TamperEngine.apply_attack_to_doc(doc, attack_type)
            return doc, 1.0
                
        except:
            return fitz.open(), 0.0
    
    @staticmethod
    def apply_attack_to_doc(doc, attack_type=None):
        if attack_type is None:
            attack_type = random.choice(['meta', 'annot', 'page', 'link', 'text', 'embed', 'layer', 'code'])
        
        try:
            if attack_type == 'meta':
                doc.set_metadata({
                    "Author": "",
                    "Title": "",
                    "Subject": ""
                })
            
            elif attack_type == 'annot':
                if doc.page_count > 0:
                    page = doc[0]
                    for i in range(3):
                        page.add_text_annot((100 + i*20, 100 + i*20), f"Comment {i}")
            
            elif attack_type == 'page':
                doc.insert_page(-1)
                doc.insert_page(-1)
            
            elif attack_type == 'link':
                if doc.page_count > 0:
                    page = doc[0]
                    for i in range(5):
                        page.insert_link({
                            "kind": fitz.LINK_URI,
                            "from": fitz.Rect(i*50, i*50, i*50+100, i*50+100),
                            "uri": f"http://malicious-site-{i}.com"
                        })
            
            elif attack_type == 'text':
                if doc.page_count > 0:
                    page = doc[0]
                    hidden_text = "CONFIDENTIAL " * 100
                    page.insert_text((0, 0), hidden_text, color=(1, 1, 1))
            
            elif attack_type == 'embed':
                fake_binary = b"MZ" + b"X" * 5000
                doc.embfile_add("payload.exe", fake_binary)
                doc.embfile_add("malware.dll", b"Y" * 3000)
            
            elif attack_type == 'layer':
                if doc.page_count > 0:
                    doc.add_ocg("HiddenLayer1", on=False)
                    doc.add_ocg("HiddenLayer2", on=False)
            
            elif attack_type == 'code':
                js_code = "<JavaScript>app.alert('Compromised'); app.launchURL('http://evil.com');</JavaScript>"
                doc.set_xml_metadata(js_code)
        
        except:
            pass
        
        return doc

    @staticmethod
    def create_dummy_pdf():
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Synthetic Training Data Sample")
        return doc