import PyPDF2

class PDFProcessor:
    def __init__(self):
        pass

    def load_pdf(self, file_path):
        """Load a PDF and return segmented text."""
        content = self.read_pdf(file_path)
        return self.segment_text(content) if content else []

    def read_pdf(self, file_path):
        """Read a PDF file and return its content"""
        content = ""
        
        # Open PDF and create PdfReader
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Add text of every page if text is present
                for i in range(len(reader.pages)):
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        content += page_text + '\n'   
                
        except FileNotFoundError:
            print(f'The file "{file_path} was not found."')
        except PyPDF2.errors.PdfReadError as e:
            print(f'Erorr reading the PDF file: {e}')     
        
        return content
    
    
    def segment_text(self, text, max_chunk_size = 256):
        """Segment the loaded text into different parts.
        Important: The segment chunk size should be that context window / max token length of the
        vector embedding model (see "vector_db.py" for model)
        """
        text = text.replace('\n', '')
        sentences = [t + '.' for t in text.split('.')]

        segments = []
        current_segment = ""
        for sentence in sentences:
            if len(current_segment.split()) + len(sentence.split()) <= (max_chunk_size // 1.5):
                current_segment += sentence
            else:
                segments.append(current_segment)
                current_segment = ""
        segments.append(current_segment)

        return segments
