import numpy as np
from PIL import Image, ImageFont, ImageDraw 
import cv2


# letters = "0123456789ACEFHPUY" + "0123456789ACEFHPUY"
letters = "0123456789" + "ACEHNPRSTVXZ"
# without B D F G I J K L M O Q U W Y
left_letters = len(letters)
letters += letters
num_letters = len(letters)
right_letters = num_letters - left_letters
print(left_letters, right_letters)

# block_margin = 20
# block_size = 60
# half_block_size = (block_size, block_size // 2)
# digit_margin = 0
# font_height = 40
# text_top = 0
# underline = 52
# font = ImageFont.truetype(r'OCRA.TTF', font_height)

# 60 set consolab
block_margin = 20
block_size = 60
half_block_size = (block_size, block_size // 2)
digit_margin = 2
font_height = 40
text_top = 12
underline = 50
font = ImageFont.truetype(r'CONSOLAB.TTF', font_height)


def prepare_digit_patches(background_color:int, foreground_color:int):
    left_digits = []
    right_digits = []
    
    for letter in letters:
        # left
        block = Image.fromarray((np.ones(half_block_size) * background_color).astype(np.uint8))
        draw = ImageDraw.Draw(block)
        bbox = font.getbbox(letter)
        
        x = half_block_size[1] - bbox[2] - digit_margin
        y = text_top
        draw.text((x, y), letter, fill=foreground_color, font=font)
        draw.line([(digit_margin + 4, underline), (half_block_size[1], underline)], fill=foreground_color, width=5)
        left_digits.append(np.asarray(block))
        
        
        # right
        block = Image.fromarray((np.ones(half_block_size) * background_color).astype(np.uint8))
        draw = ImageDraw.Draw(block)
        bbox = font.getbbox(letter)    
        
        x = digit_margin
        y = text_top
        draw.text((x, y), letter, fill=foreground_color, font=font)
        
        right_digits.append(np.asarray(block))

    return left_digits, right_digits

    
# left_digits_b, right_digits_b = prepare_digit_patches(background_color=0, foreground_color=255)  # 黑底白字
left_digits_w, right_digits_w = prepare_digit_patches(background_color=255, foreground_color=0)  # 白底黑字

h, w = 2970, 2100
batch = 4 * 11  # batch rows for each suit

def prepare_checker_digits(left_digits, right_digits, digit_is_black_on_white:bool):    
    img = (np.ones((h, w))*(0 if digit_is_black_on_white else 255)).astype(np.uint8)

    cnt = 0
    sep = block_size//2
    for x in range(0, h - block_size + 1, block_size):
        for y in range(0, w - block_size + 1, block_size):
            x_, y_ = x // block_size, y // block_size

            if (x_ + y_) % 2 == 0:
                continue
            
            block = np.ones((block_size, block_size))*(255 if digit_is_black_on_white else 0)

            if x_ % 4 == 0:
                img[x:x+block_size,y:y+block_size] = block
                continue

            if (x_ % batch) // 4 in [0,1,2,3] and y_ in [0,9,18,27,34]:
                img[x:x+block_size,y:y+block_size] = block
                continue

            if (x_ % batch) // 4 in [4,5,6,7]  and y_ in [0,11,21,28,34]: 
                img[x:x+block_size,y:y+block_size] = block
                continue

            if (x_ % batch) // 4 in [8] and y_ in [0,10,20,27,34]: 
                img[x:x+block_size,y:y+block_size] = block
                continue

            if (x_ % batch) // 4 in [9,10] and (y_ in [0,7,14,21,28,34] or y_ >= 28):
                img[x:x+block_size,y:y+block_size] = block
                continue

            block[:, :sep] = left_digits[cnt % len(left_digits)]
            block[:, sep:] = right_digits[(cnt // len(left_digits)) % len(right_digits)]
            img[x:x+block_size,y:y+block_size] = block
            
            cnt += 1
            # print(cnt)

            if cnt == left_letters * right_letters - 2:
                cnt = 0
            
    return img


# img = prepare_patched_digits(left_digits_b[:left_letters], right_digits_b[num_non_digit_letters:], background_color=255)
img = prepare_checker_digits(left_digits_w[:left_letters], right_digits_w[left_letters:], digit_is_black_on_white=True)

img = Image.fromarray(img)
img.save('pattern.png')

