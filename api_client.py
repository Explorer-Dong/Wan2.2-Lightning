import json
import requests
import tqdm
import time

class GenerateClient:
    """通用生成任务客户端，支持 t2v 和 i2v 请求"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        初始化客户端
        :param host: 服务IP地址
        :param port: 服务端口
        """
        self.base_url = f"http://{host}:{port}"
        self.headers = {"Content-Type": "application/json"}

    def _post(self, endpoint: str, payload: dict) -> dict:
        """内部方法：发送POST请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Error] POST {url} failed:", e)
            return {"error": str(e)}

    def generate(self, service_type: str, prompt: str, image_url: str = None) -> dict:
        """
        通用生成接口
        :param service_type: 任务类型（t2v 或 i2v）
        :param prompt: 文本提示
        :param image_url: 图片URL（仅i2v使用）
        """
        payload = {"service_type": service_type, "prompt": prompt}
        if image_url:
            payload["image_url"] = image_url

        return self._post("/generate", payload)

    def generate_t2v(self, prompt: str) -> dict:
        """文本转视频 (t2v)"""
        return self.generate("t2v", prompt)

    def generate_i2v(self, prompt: str, image_url: str) -> dict:
        """图像转视频 (i2v)"""
        return self.generate("i2v", prompt, image_url)


def simulate_request(task: str, prompt_list: list, image_list: list | None):
    for i in range(len(prompt_list)):
        # 发起请求
        if task == 't2v':
            response = client.generate_t2v(
                prompt=prompt_list[i]
            )
        else:
            response = client.generate_i2v(
                prompt=prompt_list[i],
                image_url=image_list[i]
            )
        
        # 打印
        print(response)


if __name__ == "__main__":
    client = GenerateClient("127.0.0.1", 8000)

    t2v_prompt_list = [
        'A serene landscape featuring a large, lush tree with drooping branches, situated on a small island in a calm lake. The scene is illuminated by soft sunlight, with fluffy clouds in a bright blue sky and green foliage surrounding the water. make a video.',
        'A cheerful corgi wearing sunglasses lounges on a bright orange flotation device, floating in a sparkling blue ocean. The corgi gently bobs up and down with the movement of the waves under a sunny sky with fluffy white clouds. make a video.',
        'A young woman with long blonde hair, wearing a white t-shirt and blue jeans, walking through a sunny park with green trees in the background, carrying a brown shoulder bag, followed by a smooth tracking shot that moves alongside her as she walks. make a video.',
        'A ballerina in a flowing silver dress dancing on a deserted urban rooftop at night, city skyline twinkling behind, spotlight on her, camera crane shot rising, ultra detailed, cinematic. make a video.',
        'A cozy coffee cup on a cafe table by the window, with the view of the street outside — camera zooms out from the coffee cup, capturing the cozy interior and view of the street outside. make a video.',
    ]

    i2v_prompt_list = [
        '''
        A dynamic, cinematic action shot, captured from a low frontal angle, depicting a central subject riding on the back of a powerfully sprinting tiger at full speed through a dense, mist-shrouded forest. The tiger is in a powerful mid-gallop, muscles straining, with its front paws thrust dramatically toward the lens, creating a intense sense of speed and perspective. The subject leans forward dynamically, gripping the tiger's fur, with hair swept back sharply by the wind.
        Crucially, the subject's facial features, expression, and hairstyle must maintain 95%+ consistency with the original reference image, with zero facial distortion or loss of detail. The character's likeness must be preserved perfectly.
        The background is a blur of tall, slender trees receding into blue-gray mist, with pronounced motion blur emphasizing the high velocity. Soft, diffused daylight filters through the canopy, evenly illuminating the scene. The image is rendered in high-resolution, cinematic clarity, perfectly blending a sharply detailed foreground with a soft background to convey raw speed, authentic scale, and a harmonious bond between rider and beast.
        ''',
        '''
        Create a cinematic video scene based on the uploaded subject image.
        - Foreground (Runner / Survivor): The uploaded subject, fully recognizable (≥95% similarity), running forward in fear, urgency, or panic, with dynamic arm and leg motion and expressive body language that conveys tension.
        The motion should be smooth and continuous across frames, maintaining physical realism and emotional coherence.
        - Background / Pursuing Character (Horror Hunter): A fixed horror-style Identity V Hunter, terrifying and grotesque in appearance, chasing the subject.
        The Hunter starts from a distance and gradually approaches the subject over time, creating a suspenseful chase dynamic.
        All movements should feel physically grounded, realistic, and continuous.
        - Environment: An Identity V–inspired gothic or abandoned map, featuring eerie streets, courtyards, and decayed gothic buildings, surrounded by atmospheric mist or haze to sustain a cinematic, tense ambiance.
        - Lighting: Cinematic diffused lighting with soft shadows and volumetric haze, highlighting textures of clothing, environment, and the Hunter.
        Lighting consistency across frames is required to ensure visual realism.
        - Composition: Mid-shot framing, keeping both the subject (runner) and Hunter visible throughout the chase, with clear spatial depth and linear perspective.
        - Style: Realistic cinematic rendering blended with Identity V’s gothic aesthetic, emphasizing natural textures, photorealistic surfaces and fabrics, atmospheric haze, horror tension, and dynamic movement.
        The video should gradually increase tension, showing the subject becoming more threatened as the Hunter closes in.
        ''',
        '''
        Generate a short video based on the uploaded animal image. The subject’s species, anatomy, proportions, and natural features must remain completely unchanged — preserve the original face shape, fur texture, body type, and overall realism. No anthropomorphism, deformation, or species blending. Maintain 95% consistency in composition, lighting, and style with the source image.
        Scene:
        Keep the same background, camera angle, and lighting as the uploaded image. The animal stays in position, naturally lit, with realistic surface texture and fine detail.
        Action:
        The animal slowly looks left and right, moving its eyes and head with subtle, natural motion. Occasionally blinks or tilts the head slightly, appearing somewhat nervous or uneasy. The movements should feel organic, continuous, and true to the species’ real behavior — no exaggerated or cartoon motion.
        Style & Mood:
        Realistic lighting, stable camera, warm and natural tone, minimal humor, with a subtle sense of awkwardness or curiosity. Smooth motion, gentle rhythm.
        Requirements:
        - Preserve the original animal subject fully;
        - No human-like features, no morphing, no stylization;
        - Maintain consistent background, lighting, and perspective;
        - 95% style and detail match to the original still image;
        - Realistic and coherent motion sequence.
        ''',
        '''
        Create a dynamic video based on the uploaded image.

        Keep the uploaded animal completely unchanged — same species, face, body shape, and appearance. 
        Do NOT morph or replace it into a cat, dog, or any other animal.

        Animate the scene so that the animal, wearing a black and yellow bee costume with small transparent wings, starts to move slightly, then spreads its wings and gently takes off into the air. 
        The wings flap gracefully with glowing reflections, carrying the animal upward in a smooth, magical motion. 
        Its expression remains curious and calm as it rises.

        As the animal lifts off, soft particles of glowing dust or pollen drift through the air, creating a dreamy, cinematic atmosphere. 
        The background stays softly blurred with warm lighting, evoking a cozy and magical feeling of flight within an indoor space.

        Lighting: warm, soft, cinematic glow.  
        Motion: fluid, natural, visually striking.  
        Style: realistic, detailed, magical realism.

        Negative prompts: species transformation, distorted motion, unnatural flight, overexposure, cartoon style, low detail, exaggerated physics.
        ''',
        '''
        Create a photorealistic, cinematic-style video featuring the uploaded animal as the main subject.  
        Keep the animal’s true species, color, fur texture, facial features, and anatomy completely unchanged.  
        Replace the animal in the reference image while maintaining 95% similarity in pose, composition, lighting, and style.  
        Scene setup:  
        The animal stands or jumps on a stage, holding or interacting with an instrument (electric guitar or similar), in an exaggerated “rock star” pose.  
        Expression: intense, energetic, playful, with spiked or tousled fur/ears/hat enhancing the rock vibe.  
        Video motion:  
        - Big, dynamic movements: head banging, front paws strumming or clawing the instrument, tail swaying, ears flapping.  
        - Camera transitions: multiple angles including eye-level, low-angle, slight top-down, and side sweeps.  
        - Smooth cuts or pans between different perspectives to create dynamic stage storytelling.  
        - Stage lights: multicolored beams, flashes, and sweeps, synchronized with the pet’s energetic movements.  
        - Environmental effects: smoke, subtle reflections on the stage floor, light glows, lens flares.  
        - Duration: 8–12 seconds, continuous action, loopable if needed.  
        Background:  
        Indoor or concert-style stage with lights, slight fog, and subtle home props if applicable.  
        Bright and soft illumination on the animal while keeping dramatic multicolored stage lights in the background.  
        Style:  
        Photorealistic, cinematic realism, vibrant warm tones, soft contrast, high detail on fur, instrument, and stage props.  
        Ensure the uploaded animal remains unchanged as the main subject, naturally integrated into the scene while performing a wild, energetic rock concert with camera transitions and dynamic stage lighting.
        ''',
        '''
        Generate a cinematic video where a subject, after opening their mouth, begins to transform into a werewolf. The transformation should be intense and supernatural, while the subject’s key features remain recognizable throughout the process.
        Scene Composition & Key Frames:
        1. Starting Position (0-20% of video):
        - The subject is standing or positioned in a dramatic setting with a fierce, predatory expression.
        - Eyes glow in a supernatural yellow or red, and fangs may be partially visible.
        - Lighting is dark and moody, highlighting glowing eyes and sharp fangs.
        2. Mouth Opening (20-40% of video):
        - The subject opens their mouth wider, showing sharp, elongated fangs fully, as if about to howl or snarl.
        - Facial muscles begin to shift, eyes intensify, glowing more vividly.
        - Skin, fur, or head features start to morph, showing early werewolf traits: nose elongating, cheekbones sharpening, fur appearing along jawline, ears, or head.
        3. Full Transformation into Werewolf (40-80% of video):
        - Facial and head features morph further into a werewolf form: jawline extends, teeth elongate into wolf-like fangs.
        - Fur grows more extensively on face, neck, and shoulders. Skin, scales, or fur texture becomes rougher and more primal.
        - Neck, shoulders, or body may start changing; clothing (if any) may tear naturally.
        4. Complete Werewolf Transformation (80-100% of video):
        - Subject is fully transformed into a werewolf, with glowing eyes, elongated face, fur, pointed ears, and predatory expression.
        - Final frame zooms slightly on the eyes, emphasizing fierce, intense gaze and the subject’s full werewolf nature.
        Visual Effects:
        - Fur growth should appear organic and gradual throughout the transformation.
        - Eyes remain glowing, shifting in intensity and color (yellow or red) as transformation progresses.
        - Lighting shifts to enhance the dramatic atmosphere, with moonlight or dim ambient light emphasizing the supernatural effect.
        Facial/Head Features:
        - The subject’s original facial, head, or distinguishing features should remain recognizable throughout, even as jawline, eyes, fangs, and fur transform.
        - Transformation maintains sharpness or key identity traits, whether human or animal, while showing the full werewolf form.
        '''
    ]

    i2v_image_list = [
        'https://cdn.dwj601.cn/temp/图片_002_01.png',
        'https://cdn.dwj601.cn/temp/图片_003_01.png',
        'https://cdn.dwj601.cn/temp/图片_004_01.png',
        'https://cdn.dwj601.cn/temp/图片_005_01.png',
        'https://cdn.dwj601.cn/temp/图片_006_01.png',
        'https://cdn.dwj601.cn/temp/图片_007_01.png',
    ]

    assert len(i2v_prompt_list) == len(i2v_image_list)

    simulate_request(task="t2v", prompt_list=t2v_prompt_list, image_list=None)
    simulate_request(task="i2v", prompt_list=i2v_prompt_list, image_list=i2v_image_list)
