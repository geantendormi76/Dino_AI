import sys
from pathlib import Path
import os
import random
from PIL import Image

# [核心修改] 让脚本能够感知到项目的根目录
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# --- [核心修改] 使用绝对路径定义输入和输出 ---
CONFIG = {
    "assets_path": project_root / "_assets",
    "output_path": project_root / "data" / "classification_data",
    "image_size": (100, 100),
    "background_color": (247, 247, 247),
    "images_per_class": 500,
    "min_scale": 0.7,
    "max_scale": 1.0,
}

def generate_classification_dataset():
    """
    [核心修正] 直接、忠实地使用 _assets 文件夹中的每一个原始 png 文件来生成数据。
    """
    assets_path = Path(CONFIG["assets_path"])
    output_path = Path(CONFIG["output_path"])

    # 确保输出目录存在
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"输入素材目录: {assets_path}")
    print(f"输出数据集目录: {output_path}")

    if not assets_path.exists():
        print(f"❌ 错误：资产文件夹 '{assets_path}' 不存在！")
        return

    asset_files = list(assets_path.glob("*.png"))
    if not asset_files:
        print(f"❌ 错误：在 '{assets_path}' 中没有找到任何.png文件！")
        return

    print(f"🔍 找到了 {len(asset_files)} 个资产文件。开始生成数据集...")

    for asset_file in asset_files:
        if 'dino-player' in asset_file.stem:
            print(f"  -> 跳过非障碍物素材: {asset_file.name}")
            continue

        class_name = asset_file.stem
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"  -> 正在为类别 '{class_name}' 生成 {CONFIG['images_per_class']} 张图片...")
        
        try:
            asset_img = Image.open(asset_file).convert("RGBA")
        except Exception as e:
            print(f"   ⚠️ 警告：无法打开文件 {asset_file}，已跳过。错误: {e}")
            continue

        for i in range(CONFIG["images_per_class"]):
            background = Image.new("RGBA", CONFIG["image_size"], CONFIG["background_color"])
            
            max_allowed_scale_w = CONFIG["image_size"][0] / asset_img.width
            max_allowed_scale_h = CONFIG["image_size"][1] / asset_img.height
            final_max_scale = min(max_allowed_scale_w, max_allowed_scale_h, CONFIG["max_scale"])
            
            if final_max_scale < CONFIG["min_scale"]:
                scale = final_max_scale
            else:
                scale = random.uniform(CONFIG["min_scale"], final_max_scale)

            new_width = int(asset_img.width * scale)
            new_height = int(asset_img.height * scale)
            
            if new_width <= 0 or new_height <= 0: continue

            resized_asset = asset_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            max_x = CONFIG["image_size"][0] - new_width
            max_y = CONFIG["image_size"][1] - new_height
            
            paste_x = random.randint(0, max(0, max_x))
            paste_y = random.randint(0, max(0, max_y))

            background.paste(resized_asset, (paste_x, paste_y), resized_asset)
            
            final_image = background.convert("RGB")
            output_file_path = class_dir / f"{i:04d}.png"
            final_image.save(output_file_path)

    print(f"\n✅ 数据集生成完毕！已保存至 '{output_path}' 目录。")

if __name__ == "__main__":
    generate_classification_dataset()