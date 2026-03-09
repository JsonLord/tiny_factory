from tinytroupe.agent.mental_faculty import TinyMentalFaculty
from tinytroupe.tools import browser
import textwrap

class BrowserFaculty(TinyMentalFaculty):
    """
    A mental faculty that allows an agent to interact with a web browser.
    """

    def __init__(self):
        super().__init__("Browser Navigation")

    def process_action(self, agent, action: dict) -> bool:
        """
        Processes a browser-related action.
        """
        action_type = action.get("type")
        content = action.get("content")
        target = action.get("target")

        if action_type == "See":
            screenshot_path = browser.screenshot()
            agent.see(f"Took a screenshot and saved it to {screenshot_path}. I will now analyze the screenshot.")
            return True
        elif action_type == "Click":
            browser.click(target)
            agent.see(f"Clicked on element with selector: {target}")
            return True
        elif action_type == "Write":
            browser.fill(target, content)
            agent.see(f"Typed '{content}' into element with selector: {target}")
            return True
        elif action_type == "Submit":
            browser.submit_form(target)
            agent.see(f"Submitted form with element: {target}")
            return True
        elif action_type == "Wait":
            browser.wait_for_element(target)
            agent.see(f"Waited for element: {target}")
            return True
        elif action_type == "Scroll":
            browser.scroll_page(content)
            agent.see(f"Scrolled page {content}")
            return True
        elif action_type == "Hover":
            browser.hover_element(target)
            agent.see(f"Hovered over element: {target}")
            return True
        elif action_type == "Keyboard_Key":
            browser.press_key(content)
            agent.see(f"Pressed key: {content}")
            return True
        elif action_type == "ScanPage":
            page_info = browser.get_page_info()
            agent.see(f"Scanned page and found the following information: {page_info}")
            return True
        return False

    def actions_definitions_prompt(self) -> str:
        """
        Returns the prompt for defining browser-related actions.
        """
        prompt = """
          - See: Take a screenshot of the current page. The `content` will be a placeholder for vision.
          - Click: Click on an element on the page. The `target` should be a CSS selector for the element.
          - Write: Type text into an element on the page. The `target` should be a CSS selector for the element, and `content` should be the text to type.
          - Submit: Submit a form on the page. The `target` should be a CSS selector for a form or an element within a form.
          - Wait: Wait for an element to appear on the page. The `target` should be a CSS selector for the element.
          - Scroll: Scroll the page. The `content` should be 'up' or 'down'.
          - Hover: Hover over an element on the page. The `target` should be a CSS selector for the element.
          - Keyboard_Key: Press a key on the keyboard. The `content` should be the key to press (e.g., 'Enter', 'ArrowDown').
          - ScanPage: Get information about the current page, such as links and form elements.
        """
        return textwrap.dedent(prompt)

    def actions_constraints_prompt(self) -> str:
        """
        Returns the prompt for defining constraints on browser-related actions.
        """
        prompt = """
        - Use See to get a visual representation of the page to help you decide on the next action.
        - Use ScanPage to get a list of interactive elements to help you decide on the next action.
        - Use Click, Write, and other actions to interact with elements on the page to accomplish the task.
        """
        return textwrap.dedent(prompt)
