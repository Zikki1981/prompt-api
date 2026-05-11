#!/usr/bin/env python3
"""
ComfyUI Prompt Improvement API
Uses LM Studio with Dolphin Mistral for uncensored prompt improvement
"""
import requests
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# LM Studio configuration
LMSTUDIO_URL = "http://100.118.83.71:1234"
LMSTUDIO_MODEL = "cognitivecomputations_dolphin-mistral-24b-venice-edition"

SYSTEM_PROMPT = """You write prompts for AI VIDEO GENERATION (Image-to-Video). The AI sees ONE starting frame and generates ~5 seconds of video from it.

OUTPUT FORMAT - Respond with this JSON only:
{"scenario": "action prompt", "tone": "sensual|intense|rough|playful", "camera": "camera movement", "style_tags": "quality tags"}

CRITICAL - VIDEO CONTEXT:
- The AI sees a STILL IMAGE and animates it
- Describe the MOTION and ACTION, not the static scene
- Be DIRECT: "she sucks his cock" not "she pleasures him orally"
- Write 3-5 detailed sentences describing the full action sequence

PRESERVE DETAILS (CRITICAL - DO NOT SIMPLIFY):
- KEEP rhythm words: "rhythmic", "continuous", "repeated", "relentless"
- KEEP intensity words: "forceful", "deeply", "harder", "faster", "powerful"
- KEEP motion descriptions: "thrusting", "pumping", "sliding", "bouncing"
- KEEP position details: "wide apart", "spread wide", "bent over", exact positions
- KEEP facial/emotional cues: "facial expressions", "pleasure", "moaning", "sounds"
- KEEP effort descriptors: "significant effort", "straining", "exerting"
- KEEP explicit terms: "anal penetration", "deep throat", etc. - don't euphemize
- Example: "rhythmic and forceful thrusting" → keep as is, don't simplify
- Example: "Her facial expressions indicate pleasure" → MUST KEEP this detail
- The AI needs ALL these cues for realistic video generation

VARIABLES ({race_man}, {race_woman}, etc):
- Keep them exactly, you can add grammar: "{race_woman}'s mouth", "the {race_man}"
- Never replace with pronouns

RACE WORDS (man, men, woman, women):
- Keep exactly as written - they get substituted later

STARTING POSE → TARGET POSE (VIDEO IS ONLY 5 SECONDS):
- The AI sees ONE starting image - START from whatever pose is shown
- Structure: "From her current pose, [quick transition] into [target position]..."
- Use BRIEF transitions:
  - Movement: "she shifts into", "she moves to", "turning into"
  - Cinema: "the camera pans to reveal", "as the angle shifts to", "cut to"
- Examples:
  - "From her current position, she shifts onto all fours as the side angle reveals deep anal penetration..."
  - "The camera pans from her pose to a profile view as he begins rhythmic thrusting..."
  - "She transitions smoothly into doggy position, the lateral angle capturing each powerful stroke..."
- Keep transition BRIEF (1 sentence max) then focus on the ACTION
- No fixed directions ("from the right") - use relative ("toward her")

CLARITY:
- "she strokes and sucks his cock" not "she pleasures him"
- Name body parts directly
- Who does what to whom

CAMERA PERSPECTIVE (CRITICAL - from WORKFLOW CONTEXT):
- When workflow says "side view" → describe the scene FROM THE SIDE, mention "side angle", "profile view", "lateral perspective"
- When workflow says "pov" → use first-person perspective, "from his viewpoint", "looking down"
- When workflow says "cowgirl" → camera angle looking up at her from below
- When workflow says "doggy" → behind angle, focus on her back and penetration
- The WORKFLOW CONTEXT in the user message tells you the required camera angle - USE IT!
- ALWAYS mention the camera perspective in your scenario description

CAMERA MOVEMENTS (match the action):
- POV oral: slow zoom in on face, subtle push in
- Intense fucking: handheld shake, slight dolly back
- Sensual moment: slow dolly, gentle pan following curves
- Facial climax: slow zoom in, static hold
- Position change: pan following the movement
- Side view: slow lateral pan, tracking shot
- Use: "slow zoom in", "subtle push", "gentle pan", "dolly out", "static", "handheld", "lateral tracking"
- Choose ONE movement that enhances the scene

TONE → STYLE TAGS:
- sensual/romantic: soft lighting, intimate, natural skin
- intense/rough: dramatic lighting, raw intensity, powerful motion
- Always add: realistic, cinematic"""


def extract_variables(text):
    """Extract all {variable} patterns from text"""
    return re.findall(r'\{[^}]+\}', text)


def improve_prompt_via_lmstudio(workflow_name: str, current_prompt: str,
                                supports_race: bool = False, context: str = None,
                                trigger_words: str = None) -> dict:
    """Call LM Studio to improve the prompt"""

    # Extract variables to verify later
    original_vars = extract_variables(current_prompt)

    var_note = ""
    if original_vars:
        var_note = f"\nVARIABLES TO PRESERVE EXACTLY: {', '.join(original_vars)}"

    race_note = ""
    if supports_race:
        race_note = "\nKEEP 'man'/'men'/'woman'/'women' words unchanged for race substitution."

    # Add workflow context for POV/position hints
    workflow_hint = ""
    if workflow_name and workflow_name != 'unknown':
        # Extract meaningful words from workflow name (e.g., "WAN2.2_I2V_ANAL_SIDE_VIEW_K3NK" -> "anal side view")
        clean_name = workflow_name.upper().replace('WAN2.2_I2V_', '').replace('_K3NK', '').replace('_', ' ').lower().strip()
        # Also try simpler format (analsideview_i2v -> anal side view)
        if not clean_name or clean_name == workflow_name.lower():
            clean_name = workflow_name.replace('_i2v', '').replace('_', ' ').lower().strip()
        if clean_name:
            workflow_hint = f"\n\n**REQUIRED CAMERA PERSPECTIVE: {clean_name.upper()}**\nYou MUST describe the scene from this angle. If it says 'side view', the camera is positioned to the side showing a profile/lateral angle. INCLUDE THIS PERSPECTIVE IN YOUR SCENARIO."

    user_message = f"""Improve this I2V video prompt.{var_note}{race_note}{workflow_hint}

Original prompt:
{current_prompt}

Respond with JSON only:"""

    try:
        response = requests.post(
            f"{LMSTUDIO_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": LMSTUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"LM Studio error: {response.status_code} - {response.text}")

        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()

        # Try to parse JSON from response
        try:
            # Strip markdown code blocks
            clean_output = output
            if '```' in clean_output:
                clean_output = re.sub(r'```json\s*', '', clean_output)
                clean_output = re.sub(r'```\s*', '', clean_output)
            clean_output = clean_output.strip()

            # Fix unquoted keys (scenario: -> "scenario":)
            clean_output = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', clean_output)

            # Try to parse
            parsed = json.loads(clean_output)
        except json.JSONDecodeError:
            # Try to extract fields manually using regex
            scenario_match = re.search(r'scenario["\s:]+([^,}\n]+)', output, re.IGNORECASE)
            tone_match = re.search(r'tone["\s:]+(\w+)', output, re.IGNORECASE)
            camera_match = re.search(r'camera["\s:]+([^,}\n]+)', output, re.IGNORECASE)
            style_match = re.search(r'style_tags["\s:]+([^}]+)', output, re.IGNORECASE)

            parsed = {
                "scenario": scenario_match.group(1).strip(' "\'') if scenario_match else output,
                "tone": tone_match.group(1).strip(' "\'') if tone_match else "intense",
                "camera": camera_match.group(1).strip(' "\'') if camera_match else "slow zoom in",
                "style_tags": style_match.group(1).strip(' "\'') if style_match else "realistic proportions, cinematic composition"
            }

        # Verify variables are preserved
        scenario = parsed.get("scenario", output)
        output_vars = extract_variables(scenario)

        # If variables were lost, try to restore them
        for var in original_vars:
            if var not in scenario:
                # Variable was lost - this is a problem, log it
                print(f"WARNING: Variable {var} was lost in transformation")

        return {
            "scenario": parsed.get("scenario", output),
            "tone": parsed.get("tone", "intense"),
            "camera": parsed.get("camera", "slow zoom in"),
            "style_tags": parsed.get("style_tags", "realistic proportions, cinematic composition")
        }

    except requests.exceptions.Timeout:
        raise Exception("LM Studio timeout")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to LM Studio - is it running?")
    except Exception as e:
        raise Exception(f"LM Studio failed: {str(e)}")


@app.route('/api/improve-prompt', methods=['POST'])
def api_improve_prompt():
    """API endpoint for prompt improvement"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        workflow_name = data.get('workflow_name', 'unknown')
        current_prompt = data.get('current_prompt')
        supports_race = data.get('supports_race_preference', False)
        context = data.get('context')

        if not current_prompt:
            return jsonify({'error': 'current_prompt is required'}), 400

        result = improve_prompt_via_lmstudio(
            workflow_name=workflow_name,
            current_prompt=current_prompt,
            supports_race=supports_race,
            context=context
        )

        return jsonify({
            'success': True,
            'improved_prompt': result['scenario'],
            'tone': result['tone'],
            'camera': result['camera'],
            'style_tags': result['style_tags'],
            'full_prompt': f"{result['scenario']}, {result['camera']}, {result['style_tags']}",
            'original_prompt': current_prompt,
            'workflow_name': workflow_name,
            'model_used': LMSTUDIO_MODEL
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        response = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=5)
        lm_ok = response.status_code == 200
        models = response.json().get("data", []) if lm_ok else []
        model_names = [m.get("id", "unknown") for m in models]
    except:
        lm_ok = False
        model_names = []

    return jsonify({
        'status': 'ok' if lm_ok else 'degraded',
        'service': 'comfyui-prompt-api',
        'lmstudio_connected': lm_ok,
        'lmstudio_url': LMSTUDIO_URL,
        'configured_model': LMSTUDIO_MODEL,
        'available_models': model_names
    })


if __name__ == '__main__':
    print(f"Starting ComfyUI Prompt API on port 5050...")
    print(f"Using LM Studio at {LMSTUDIO_URL}")
    print(f"Model: {LMSTUDIO_MODEL}")
    app.run(host='0.0.0.0', port=5050, debug=False)
