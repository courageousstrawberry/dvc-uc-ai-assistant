import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import extract_course_list as ecl

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing. Put it in your .env file.")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    """
    # Load your course data
    with open("agreements_25-26/ucb_25-26_cs.json", "r") as file:
        data = json.load(file)

    ----------Test function:----------
    def search_courses(course_code):
        #Search for DVC courses that match the given course code.
        results = []
        for category in data.get("Berkeley", []):
            if "Courses" in category:
                for course_pair in category["Courses"]:
                    dvc_data = course_pair.get("DVC", {})
                    uc_course = course_pair.get("UC_Berkeley", {})
                    
                    # DVC can be a dict or a list of dicts
                    dvc_courses = dvc_data if isinstance(dvc_data, list) else [dvc_data]
                    
                    for dvc_course in dvc_courses:
                        if not dvc_course:
                            continue
                        dvc_code = dvc_course.get("Course_Code", "")
                        if course_code.upper() in dvc_code.upper():
                            results.append({
                                "category": category.get("Category", ""),
                                "dvc_course": dvc_course,
                                "uc_course": uc_course
                            })
        return results
    """
    # Keep conversation history for context
    conversation_history = [
        {"role": "system", "content": "You are a helpful DVC course advisor that helps students understand which courses transfer to UCs for Computer Science major."}
    ]
    print("------DVC Course Advisor: Type 'quit' or 'exit' to end the conversation------\n")

    # User query
    while True:
        try:
            print("Ask DVC Chatbot anything related to UC transfer requirements: ")
            user = input()
            print()

            if user.lower() in ['quit', 'exit']:
                print("Goodbye! Good luck transferring!")
                break
            if not user:
                continue
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user})
            
            # Give chatbot some context - could also implement key words
            context = f"Here are the matching DVC courses that transfer to each corresponding UC: {ecl.ucd_map}, {ecl.uci_data}, {ecl.ucd_data}, {ecl.ucsd_data}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content

            conversation_history.append({"role": "assistant", "content": assistant_response})
            print(f"\nResponse: {assistant_response}\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye! Good luck with your transfer!")
            break
        except Exception as e:
            print("API call failed:", e)

if __name__ == "__main__":
    main()
