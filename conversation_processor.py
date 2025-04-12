import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Environment setup
load_dotenv()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini API setup
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise

class ConversationProcessor:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def process_conversation(self, text):
        """
        Takes text input and uses Gemini to separate doctor-patient conversations.
        
        Args:
            text (str): Conversation text to analyze
            
        Returns:
            dict: Results in the following format:
                {
                    'processed_text': 'Doctor: How are you feeling today?@@Patient: I haven't been sleeping well.',
                    'starts_with_doctor': True
                }
        """
        try:
            # Construct Gemini prompt
            prompt = f"""
            The text below is a conversation between a doctor and a patient. 
            Analyze this conversation to identify each speaker (doctor/patient), and output in the following format:
            
            Output format:
            1. Whether the conversation starts with the doctor (true/false)
            2. Conversation content connected with "@@" delimiter (just the conversation content without speaker labels)
            
            Input text:
            {text}
            
            Please respond in JSON format:
            {{
                "starts_with_doctor": true/false,
                "conversation": "first utterance@@second utterance@@third utterance"
            }}
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON part from response
            import json
            import re
            
            # Extract JSON format
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
                
            # Remove unnecessary characters
            json_str = re.sub(r'```json|```', '', json_str).strip()
            
            # Parse JSON
            try:
                result = json.loads(json_str)
                logger.info("Successfully processed conversation with Gemini")
                return {
                    'processed_text': result['conversation'],
                    'starts_with_doctor': result['starts_with_doctor']
                }
            except json.JSONDecodeError:
                # Manual parsing if JSON parsing fails
                logger.warning("Failed to parse JSON from Gemini response, attempting manual parsing")
                lines = response_text.strip().split('\n')
                starts_with_doctor = any('true' in line.lower() and 'starts_with_doctor' in line.lower() for line in lines)
                conversation_lines = []
                
                for line in lines:
                    if '@@' in line:
                        conversation_lines.append(line)
                
                processed_text = '@@'.join([l.strip() for l in conversation_lines]) if conversation_lines else text
                
                return {
                    'processed_text': processed_text,
                    'starts_with_doctor': starts_with_doctor
                }
                
        except Exception as e:
            logger.error(f"Error in processing conversation: {e}")
            # Return default values on error
            return {
                'processed_text': text,
                'starts_with_doctor': True  # Set True as default value
            }

# Test code
if __name__ == "__main__":
    processor = ConversationProcessor()
    test_text = "Hello, how are you doing? I haven't been sleeping well lately. When did these symptoms start? About 2 weeks ago."
    result = processor.process_conversation(test_text)
    print(f"Processed text: {result['processed_text']}")
    print(f"Starts with doctor: {result['starts_with_doctor']}") 