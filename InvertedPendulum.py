# تثبيت خادم العرض الوهمي
!apt-get update && apt-get install -y xvfb

# تثبيت المكتبات المعتمدة (النسخ الحديثة تدعم gymnasium)
!pip install stable-baselines3[extra] gymnasium[mujoco] pyvirtualdisplay moviepy

import os
import glob
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from pyvirtualdisplay import Display
from IPython.display import Video, display

# تشغيل الشاشة الوهمية لبيئة كاجل
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
print("تم تشغيل الشاشة الوهمية بنجاح.")

# 1. تحديد اسم البيئة
env_id = "InvertedPendulum-v4"

# 2. إنشاء بيئة التدريب
env = gym.make(env_id)

# 3. بناء نموذج PPO
# verbose=1 لعرض معلومات التدريب أثناء التنفيذ
model = PPO("MlpPolicy", env, verbose=1)

print(f"بدء تدريب الوكيل على بيئة {env_id}...")
# 4. التدريب
model.learn(total_timesteps=25000)

# 5. حفظ النموذج
model.save("ppo_inverted_pendulum")
env.close()
print("تم التدريب وحفظ النموذج.")


import gymnasium as gym
import imageio
from IPython.display import Video, display
from stable_baselines3 import PPO

# 1. إنشاء بيئة الاختبار (مع استخدام الإصدار v5)
env_test = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
state, _ = env_test.reset()
done = False
frames = []

# 2. تحميل نموذج الإنتاج الذي صنعناه
# (يفترض أنك قمت بتدريبه وحفظه باسم ppo_inverted_pendulum في الخلية السابقة)
model = PPO.load("ppo_inverted_pendulum")

# 3. حلقة التصوير
steps = 0
# أضفنا شرط steps < 1000 لأن البندول إذا كان ممتازاً قد يستمر للأبد!
while not done and steps < 1000:
    # التقاط الإطار (الصورة)
    frame = env_test.render()
    frames.append(frame)
    
    # السحر هنا: نطلب من SB3 القرار الحتمي (بدون عشوائية) بهدوء تام
    action, _states = model.predict(state, deterministic=True)
    
    # تنفيذ الحركة في البيئة
    next_state, reward, terminated, truncated, _ = env_test.step(action)
    
    done = terminated or truncated
    state = next_state
    steps += 1

env_test.close()

# 4. حفظ وعرض الفيديو
video_path = './mujoco_inverted_pendulum.mp4'

# ملاحظة هندسية: نستخدم macro_block_size=None لتجنب أخطاء أبعاد الفيديو الفردية في FFMPEG
imageio.mimsave(video_path, frames, fps=30, macro_block_size=None)
print(f"✅ تم تسجيل الأداء بنجاح! عدد الإطارات: {len(frames)}")

# عرض الفيديو
display(Video(video_path, embed=True))