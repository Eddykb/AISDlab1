import os
from math import log2
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def rle_encode(data):
    encoded_data = bytearray()
    n = len(data)
    i = 0
    while i < n:
        current_char = data[i]
        count = 1
        while i + count < n and data[i + count] == current_char and count < 127:
            count += 1
        if count > 1:
            encoded_data.append(count)
            encoded_data.append(current_char)
            i += count
        else:
            non_repeat_chars = bytearray()
            non_repeat_chars.append(current_char)
            i += 1
            while i < n and (i + 1 >= n or data[i] != data[i + 1]) and len(non_repeat_chars) < 127:
                non_repeat_chars.append(data[i])
                i += 1
            encoded_data.append(0x80 | len(non_repeat_chars))
            encoded_data.extend(non_repeat_chars)
    return bytes(encoded_data)


def rle_decode(encoded_data):
    decoded_data = bytearray()
    n = len(encoded_data)
    i = 0
    while i < n:
        control_byte = encoded_data[i]
        i += 1
        if control_byte & 0x80:
            length = control_byte & 0x7F
            decoded_data.extend(encoded_data[i:i + length])
            i += length
        else:
            count = control_byte
            if i >= n:
                break
            char = encoded_data[i]
            decoded_data.extend([char] * count)
            i += 1
    return bytes(decoded_data)


def mtf_encode(data: bytes) -> bytes:
    alphabet = bytearray(range(256))
    encoded = bytearray()
    for byte in data:
        index = alphabet.index(byte)
        encoded.append(index)
        del alphabet[index]
        alphabet.insert(0, byte)
    return bytes(encoded)


def mtf_decode(encoded_data: bytes) -> bytes:
    alphabet = bytearray(range(256))
    decoded = bytearray()
    for index in encoded_data:
        byte = alphabet[index]
        decoded.append(byte)
        del alphabet[index]
        alphabet.insert(0, byte)
    return bytes(decoded)


def bwt(data, chunk_size):
    transformed_data = bytearray()
    ind = []
    for start in range(0, len(data), chunk_size):
        chunk = data[start:start + chunk_size]
        index, encoded_chunk = transform_chunk(chunk)
        transformed_data.extend(encoded_chunk)
        ind.append(index)
    return bytes(transformed_data), ind


def transform_chunk(chunk):
    rotations = [chunk[i:] + chunk[:i] for i in range(len(chunk))]
    rotations.sort()
    original_index = rotations.index(chunk)
    encoded_chunk = bytes(rotation[-1] for rotation in rotations)
    return original_index, encoded_chunk


def bwt_decode(encoded_data, indices, chunk_size):
    restored_data = bytearray()
    position = 0
    index = 0
    while position < len(encoded_data):
        end = position + chunk_size if position + chunk_size <= len(encoded_data) else len(encoded_data)
        chunk = encoded_data[position:end]
        original_index = indices[index]
        restored_chunk = reverse_transform_chunk(original_index, chunk)
        restored_data.extend(restored_chunk)
        position = end
        index += 1
    return bytes(restored_data)


def reverse_transform_chunk(original_index, encoded_chunk):
    table = [(char, idx) for idx, char in enumerate(encoded_chunk)]
    table.sort()
    result = bytearray()
    current_row = original_index
    for _ in range(len(encoded_chunk)):
        char, current_row = table[current_row]
        result.append(char)
    return bytes(result)


class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_map(data):
    freq_map = {}
    for byte in data:
        freq_map[byte] = freq_map.get(byte, 0) + 1
    return freq_map


def build_huffman_tree(freq_map):
    nodes = [HuffmanNode(char=char, freq=freq) for char, freq in freq_map.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        nodes.append(merged)
    return nodes[0]


def build_code_table(root, code="", code_table=None):
    if code_table is None:
        code_table = {}
    if root is not None:
        if root.char is not None:
            code_table[root.char] = code
        build_code_table(root.left, code + "0", code_table)
        build_code_table(root.right, code + "1", code_table)
    return code_table


def huffman_encode(data):
    if not data:
        return b"", {}, 0
    freq_map = build_frequency_map(data)
    root = build_huffman_tree(freq_map)
    code_table = build_code_table(root)
    encoded_bits = "".join(code_table[byte] for byte in data)
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += "0" * padding
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i + 8]
        encoded_bytes.append(int(byte, 2))
    return bytes(encoded_bytes), code_table, padding


def huffman_decode(encoded_data, code_table, padding):
    if not encoded_data:
        return b""
    encoded_bits = "".join(f"{byte:08b}" for byte in encoded_data)
    encoded_bits = encoded_bits[:-padding] if padding > 0 else encoded_bits
    reverse_code_table = {code: char for char, code in code_table.items()}
    decoded_data = bytearray()
    current_code = ""
    for bit in encoded_bits:
        current_code += bit
        if current_code in reverse_code_table:
            decoded_data.append(reverse_code_table[current_code])
            current_code = ""
    return bytes(decoded_data)


def lzss_encode(data, window_size=2048, lookahead_buffer_size=16, min_match=3):
    compressed = bytearray()
    pos = 0
    while pos < len(data):
        window_start = max(0, pos - window_size)
        lookahead_end = min(pos + lookahead_buffer_size, len(data))
        best_offset = 0
        best_length = 0
        for i in range(window_start, pos):
            match_length = 0
            while (pos + match_length < lookahead_end and
                   i + match_length < pos and
                   data[i + match_length] == data[pos + match_length]):
                match_length += 1
            if match_length > best_length:
                best_length = match_length
                best_offset = pos - i
        if best_length >= min_match:
            compressed.append(1)  #флаг совпадения
            compressed.extend(best_offset.to_bytes(2, 'big'))
            compressed.extend(best_length.to_bytes(2, 'big'))
            pos += best_length
        else:
            compressed.append(0)  #флаг байт
            compressed.append(data[pos])
            pos += 1
    return bytes(compressed)


def lzss_decode(compressed):
    decompressed = bytearray()
    pos = 0
    while pos < len(compressed):
        flag = compressed[pos]
        pos += 1
        if flag == 0:
            if pos >= len(compressed):
                raise ValueError("Отсутствует байт...")
            decompressed.append(compressed[pos])
            pos += 1
        elif flag == 1:
            if pos + 4 > len(compressed):
                raise ValueError("Неполная запись совпадения...")
            offset = int.from_bytes(compressed[pos:pos + 2], 'big')
            length = int.from_bytes(compressed[pos + 2:pos + 4], 'big')
            pos += 4
            start = len(decompressed) - offset
            if start < 0:
                raise ValueError(" Смещение выходит за пределы буфера...")
            for i in range(length):
                decompressed.append(decompressed[start + i])
        else:
            raise ValueError("Неизвестный флаг...")
    return bytes(decompressed)


def lzw_encode(data: bytes, max_dict_size=4096) -> bytes:
    if not data:
        return b""
    if max_dict_size < 256 or max_dict_size > 65535:
        raise ValueError("Размер словаря LZW должен быть в диапазоне от 256 до 65535!")
    dictionary = {bytes([i]): i for i in range(256)}
    next_code = 256
    current = bytes([data[0]])
    codes = []
    for byte in data[1:]:
        symbol = bytes([byte])
        combined = current + symbol
        if combined in dictionary:
            current = combined
        else:
            codes.append(dictionary[current])
            if next_code < max_dict_size:
                dictionary[combined] = next_code
                next_code += 1
            current = symbol
    codes.append(dictionary[current])
    compressed = bytearray()
    for code in codes:
        compressed.extend(code.to_bytes(2, 'big'))
    return bytes(compressed)


def lzw_decode(compressed_data: bytes, max_dict_size=4096) -> bytes:
    if not compressed_data:
        return b""
    if len(compressed_data) % 2 != 0:
        raise ValueError("Длина выходных данных должна быть кратна 2")
    if max_dict_size < 256 or max_dict_size > 65535:
        raise ValueError("Размер словаря LZW должен быть в диапазоне от 256 до 65535!")
    codes = []
    for i in range(0, len(compressed_data), 2):
        codes.append(int.from_bytes(compressed_data[i:i + 2], 'big'))
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    first_code = codes[0]
    if first_code not in dictionary:
        raise ValueError("Первый код отсутствует в словаре...")
    previous = dictionary[first_code]
    result = bytearray(previous)
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = previous + previous[:1]
        else:
            raise ValueError(f"Некорректный код LZW: {code}")
        result.extend(entry)
        if next_code < max_dict_size:
            dictionary[next_code] = previous + entry[:1]
            next_code += 1
        previous = entry
    return bytes(result)


# СЛУЖЕБНЫЕ ФУНКЦИИ

def calculate_entropy(data):
    if not data:
        return 0
    counter = Counter(data)
    probabilities = [count / len(data) for count in counter.values()]
    entropy = -sum(p * log2(p) for p in probabilities)
    return entropy


def to_bytes(data, encoding: str = 'utf-8') -> bytes:
    if isinstance(data, bytes):
        return data
    elif isinstance(data, str):
        return data.encode(encoding)
    elif isinstance(data, (int, float)):
        return str(data).encode(encoding)
    elif isinstance(data, (list, tuple, set, dict)):
        return str(data).encode(encoding)
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {type(data)}")


def png_to_raw(image_path, output_path):
    image = Image.open(image_path)

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        image = image.convert('RGB')
    elif image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    raw_pixels = np.array(image)
    raw_data = raw_pixels.tobytes()
    with open(output_path, 'wb') as f:
        f.write(raw_data)


def prepare_input_file(input_file_path):
    lower_path = input_file_path.lower()
    if lower_path.endswith(".png"):
        raw_path = input_file_path.rsplit(".", 1)[0] + "_temp.raw"
        png_to_raw(input_file_path, raw_path)
        return raw_path
    return input_file_path

# ============================================================
#def cleanup_temp_raw(original_input_file, prepared_input_file):
#    if prepared_input_file != original_input_file and prepared_input_file.endswith("_temp.raw"):
#        if os.path.exists(prepared_input_file):
#            os.remove(prepared_input_file)
# ============================================================


def get_effective_image_params(original_png_path):
    image = Image.open(original_png_path)
    width, height = image.size
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        mode = 'RGB'
    elif image.mode not in ('RGB', 'L'):
        mode = 'RGB'
    else:
        mode = image.mode
    return width, height, mode

def raw_to_png_from_bytes(raw_data: bytes, output_path: str, width: int, height: int, mode: str):
    if mode == 'L':
        expected_size = width * height
        arr = np.frombuffer(raw_data[:expected_size], dtype=np.uint8).reshape((height, width))
        image = Image.fromarray(arr, 'L')
    elif mode == 'RGB':
        expected_size = width * height * 3
        arr = np.frombuffer(raw_data[:expected_size], dtype=np.uint8).reshape((height, width, 3))
        image = Image.fromarray(arr, 'RGB')
    else:
        raise ValueError(f"Неподдерживаемый режим изображения: {mode}")
    image.save(output_path)

def save_decoded_outputs(input_file_path, decompressed_data: bytes, decompressed_file_path: str):
    base_name, _ = os.path.splitext(decompressed_file_path)
    lower_input = input_file_path.lower()
    if lower_input.endswith(".txt"):
        txt_path = base_name + ".txt"
        with open(txt_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(decompressed_data.decode('utf-8', errors='replace'))
        print(f"Сохранён текстовый файл: {txt_path}")
    elif lower_input.endswith(".exe"):
        exe_path = base_name + ".exe"
        with open(exe_path, 'wb') as f:
            f.write(decompressed_data)
        print(f"Сохранён exe-файл: {exe_path}")
    elif lower_input.endswith(".png"):
        raw_path = base_name + ".raw"
        with open(raw_path, 'wb') as f:
            f.write(decompressed_data)
        print(f"Сохранён raw-файл: {raw_path}")
        width, height, mode = get_effective_image_params(input_file_path)
        preview_path = base_name + "_preview.png"
        raw_to_png_from_bytes(decompressed_data, preview_path, width, height, mode)
        print(f"Сохранён просмотр PNG: {preview_path}")




def files():
    print(
        'Выберете файл для сжатия\n'
        '1 - enwik7\n'
        '2 - текст на русском\n'
        '3 - exe файл\n'
        '4 - чб изображение\n'
        '5 - изображение в серых оттенках\n'
        '6 - цветное изображение\n'
    )
    number = int(input())
    print('Ваш выбор:', number)
    if number == 1:
        return "enwik7.txt", "enwik7"
    if number == 2:
        return "lyrics.txt", "lyrics"
    if number == 3:
        return "Ultra.exe", "exe"
    if number == 4:
        return "bw.png", "bw"
    if number == 5:
        return "grey.png", "grey"
    if number == 6:
        return "color.png", "color"
    raise ValueError("Неверный выбор файла...")

def get_file_path(base_path, file_name):
    return f"{base_path}\\{file_name}"


# КОМПРЕССОРЫ

def compressor_huffman(input_file_path, compressed_file_path, decompressed_file_path):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    compressed_data, t, pad = huffman_encode(original_data)

    with open(compressed_file_path, 'wb') as file:
        file.write(compressed_data)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    decompressed_data = huffman_decode(read_compressed_data, t, pad)

    with open(decompressed_file_path, 'wb') as file:
        file.write(decompressed_data)
    save_decoded_outputs(input_file_path, decompressed_data, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == decompressed_data
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_rle(input_file_path, compressed_file_path, decompressed_file_path):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    compressed_data = rle_encode(original_data)

    with open(compressed_file_path, 'wb') as file:
        file.write(compressed_data)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    decompressed_data = rle_decode(read_compressed_data)

    with open(decompressed_file_path, 'wb') as file:
        file.write(decompressed_data)
    save_decoded_outputs(input_file_path, decompressed_data, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == decompressed_data
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Размер после декодирования: {len(decompressed_data)}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)    


def compressor_bwt_rle(input_file_path, compressed_file_path, decompressed_file_path):
    ch_size = 1024
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    compressed_data1, indexes = bwt(original_data, ch_size)
    compressed_data2 = rle_encode(compressed_data1)

    with open(compressed_file_path, 'wb') as file:
        file.write(compressed_data2)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    decompressed_data1 = rle_decode(read_compressed_data)
    decompressed_data2 = bwt_decode(decompressed_data1, indexes, ch_size)

    with open(decompressed_file_path, 'wb') as file:
        file.write(decompressed_data2)
    save_decoded_outputs(input_file_path, decompressed_data2, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(compressed_data2)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == decompressed_data2
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_bwt_mtf_ha(input_file_path, compressed_file_path, decompressed_file_path):
    ch_size = 1024
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    compressed_data1, indexes = bwt(original_data, ch_size)
    compressed_data2 = mtf_encode(compressed_data1)
    compressed_data3, t, p = huffman_encode(compressed_data2)

    with open(compressed_file_path, 'wb') as file:
        file.write(compressed_data3)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    decompressed_data1 = huffman_decode(read_compressed_data, t, p)
    print('check1:', decompressed_data1 == compressed_data2)
    decompressed_data2 = mtf_decode(decompressed_data1)
    print('check2:', decompressed_data2 == compressed_data1)
    decompressed_data3 = bwt_decode(decompressed_data2, indexes, ch_size)
    print('check3:', decompressed_data3 == original_data)

    with open(decompressed_file_path, 'wb') as file:
        file.write(decompressed_data3)
    save_decoded_outputs(input_file_path, decompressed_data3, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(compressed_data3)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == decompressed_data3
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_bwt_mtf_rle_ha(input_file_path, compressed_file_path, decompressed_file_path):
    ch_size = 1024
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    compressed_data1, indexes = bwt(original_data, ch_size)
    compressed_data2 = mtf_encode(compressed_data1)
    compressed_data3 = rle_encode(compressed_data2)
    compressed_data4, t, p = huffman_encode(compressed_data3)

    with open(compressed_file_path, 'wb') as file:
        file.write(compressed_data4)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    decompressed_data1 = huffman_decode(read_compressed_data, t, p)
    decompressed_data2 = rle_decode(decompressed_data1)
    decompressed_data3 = mtf_decode(decompressed_data2)
    decompressed_data4 = bwt_decode(decompressed_data3, indexes, ch_size)

    with open(decompressed_file_path, 'wb') as file:
        file.write(decompressed_data4)
    save_decoded_outputs(input_file_path, decompressed_data4, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(compressed_data4)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == decompressed_data4
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_lzss(input_file_path, compressed_file_path, decompressed_file_path):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    en_data1 = lzss_encode(original_data)

    with open(compressed_file_path, 'wb') as file:
        file.write(en_data1)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    de_data1 = lzss_decode(read_compressed_data)

    with open(decompressed_file_path, 'wb') as file:
        file.write(de_data1)
    save_decoded_outputs(input_file_path, de_data1, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(en_data1)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == de_data1
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    print(f"Размер после декодирования: {len(de_data1)}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_lzss_ha(input_file_path, compressed_file_path, decompressed_file_path):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    en_data1 = lzss_encode(original_data)
    en_data2, t, p = huffman_encode(en_data1)

    with open(compressed_file_path, 'wb') as file:
        file.write(en_data2)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    de_data1 = huffman_decode(read_compressed_data, t, p)
    de_data2 = lzss_decode(de_data1)

    with open(decompressed_file_path, 'wb') as file:
        file.write(de_data2)
    save_decoded_outputs(input_file_path, de_data2, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(en_data2)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == de_data2
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    print(f"Размер после декодирования: {len(de_data2)}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_lzw(input_file_path, compressed_file_path, decompressed_file_path, max_dict_size=4096):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as f1:
        orig_data = f1.read()
    en_data = lzw_encode(orig_data, max_dict_size)

    with open(compressed_file_path, 'wb') as f2:
        f2.write(en_data)

    orig_size = len(orig_data)
    compressed_size = os.path.getsize(compressed_file_path)
    ratio = orig_size / compressed_size if compressed_size != 0 else 0

    with open(compressed_file_path, 'rb') as f3:
        compressed_data = f3.read()

    de_data = lzw_decode(compressed_data, max_dict_size)

    with open(decompressed_file_path, 'wb') as output_file:
        output_file.write(de_data)
    save_decoded_outputs(input_file_path, de_data, decompressed_file_path)
    is_match = orig_data == de_data
    print(f"Исходный размер файла: {orig_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)


def compressor_lzw_ha(input_file_path, compressed_file_path, decompressed_file_path, max_dict_size=4096):
    prepared_input_file = prepare_input_file(input_file_path)
    with open(prepared_input_file, 'rb') as file:
        original_data = file.read()
    en_data1 = lzw_encode(original_data, max_dict_size)
    en_data2, t, p = huffman_encode(en_data1)

    with open(compressed_file_path, 'wb') as file:
        file.write(en_data2)
    with open(compressed_file_path, 'rb') as file:
        read_compressed_data = file.read()

    de_data1 = huffman_decode(read_compressed_data, t, p)
    de_data2 = lzw_decode(de_data1, max_dict_size)

    with open(decompressed_file_path, 'wb') as file:
        file.write(de_data2)
    save_decoded_outputs(input_file_path, de_data2, decompressed_file_path)

    original_size = len(original_data)
    compressed_size = len(en_data2)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    is_match = original_data == de_data2
    print(f"Исходный размер файла: {original_size}")
    print(f"Размер после сжатия: {compressed_size}")
    print(f"Коэффициент сжатия: {compression_ratio:.3f}")
    print(f"Декодированные данные правильные: {is_match}")
    #cleanup_temp_raw(input_file_path, prepared_input_file)



def analyze_bwt_mtf_entropy(file_path, block_sizes):
    prepared_input_file = prepare_input_file(file_path)
    with open(prepared_input_file, 'rb') as f:
        data = f.read()
    #data = data[:100000]
    print(f'размер файла: {len(data)}')
    entropy_values = []
    for chunk_size in block_sizes:
        bwt_data, ind = bwt(data, chunk_size)
        mtf_data = mtf_encode(bwt_data)
        entropy = calculate_entropy(mtf_data)
        entropy_values.append(entropy)
        print(f"Block size: {chunk_size}, Entropy: {entropy}")

        mtf_decoded = mtf_decode(mtf_data)
        bwt_decoded = bwt_decode(mtf_decoded, ind, chunk_size)
        if bwt_decoded == data:
            print(f"Block size {chunk_size}: Data restored correctly.")
        else:
            print(f"Block size {chunk_size}: Data restoration failed!")
    plt.plot(block_sizes, entropy_values, marker='o', color='orange')
    plt.xlabel('Размер блока (в байтах)')
    plt.ylabel('Энтропия')
    plt.title('Зависимость энтропии от размера блока')
    plt.grid(True)
    plt.show()


def test_lzss_compression(data, buffer_sizes):
    ratios = []
    for buffer_size in buffer_sizes:
        print(buffer_size)
        encoded_data = lzss_encode(data, 2048, buffer_size)
        ratio = len(data) / len(encoded_data) if len(encoded_data) != 0 else 0
        ratios.append(ratio)
        print(f"Размер буфера: {buffer_size}, Коэффициент сжатия: {ratio:.2f}")
    return ratios


def test_lzw_compression(data, dict_sizes):
    ratios = []
    for dict_size in dict_sizes:
        print(dict_size)
        encoded_data = lzw_encode(data, dict_size)
        ratio = len(data) / len(encoded_data) if len(encoded_data) != 0 else 0
        ratios.append(ratio)
        print(f"Размер словаря: {dict_size}, Коэффициент сжатия: {ratio:.2f}")
    return ratios



def run_selected_compressor():
    print(
        '\nВыберите компрессор:\n'
        '1 - HA\n'
        '2 - RLE\n'
        '3 - BWT + RLE\n'
        '4 - BWT + MTF + HA\n'
        '5 - BWT + MTF + RLE + HA\n'
        '6 - LZSS\n'
        '7 - LZSS + HA\n'
        '8 - LZW\n'
        '9 - LZW + HA\n'
        '10 - График энтропии BWT + MTF\n'
        '11 - График зависимости коэффициента сжатия LZSS от размера буфера\n'
        '12 - График зависимости коэффициента сжатия LZW от размера словаря'
    )
    compressor_number = int(input("Выбор компрессора: "))

    file_name, short_name = files()
    base_path = r"C:\Users\Пользователь\Desktop\aisd1"
    input_file = get_file_path(base_path, file_name)

    if compressor_number == 1:
        compressed_file = get_file_path(base_path, f"huff_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"huff_{short_name}_decoded.bin")
        compressor_huffman(input_file, compressed_file, decompressed_file)

    elif compressor_number == 2:
        compressed_file = get_file_path(base_path, f"rle_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"rle_{short_name}_decoded.bin")
        compressor_rle(input_file, compressed_file, decompressed_file)

    elif compressor_number == 3:
        compressed_file = get_file_path(base_path, f"bwt_rle_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"bwt_rle_{short_name}_decoded.bin")
        compressor_bwt_rle(input_file, compressed_file, decompressed_file)

    elif compressor_number == 4:
        compressed_file = get_file_path(base_path, f"bwt_mtf_ha_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"bwt_mtf_ha_{short_name}_decoded.bin")
        compressor_bwt_mtf_ha(input_file, compressed_file, decompressed_file)

    elif compressor_number == 5:
        compressed_file = get_file_path(base_path, f"bwt_mtf_rle_ha_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"bwt_mtf_rle_ha_{short_name}_decoded.bin")
        compressor_bwt_mtf_rle_ha(input_file, compressed_file, decompressed_file)

    elif compressor_number == 6:
        compressed_file = get_file_path(base_path, f"lzss_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"lzss_{short_name}_decoded.bin")
        compressor_lzss(input_file, compressed_file, decompressed_file)

    elif compressor_number == 7:
        compressed_file = get_file_path(base_path, f"lzss_ha_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"lzss_ha_{short_name}_decoded.bin")
        compressor_lzss_ha(input_file, compressed_file, decompressed_file)

    elif compressor_number == 8:
        compressed_file = get_file_path(base_path, f"lzw_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"lzw_{short_name}_decoded.bin")
        compressor_lzw(input_file, compressed_file, decompressed_file, 4096)

    elif compressor_number == 9:
        compressed_file = get_file_path(base_path, f"lzw_ha_{short_name}_encoded.bin")
        decompressed_file = get_file_path(base_path, f"lzw_ha_{short_name}_decoded.bin")
        compressor_lzw_ha(input_file, compressed_file, decompressed_file, 4096)

    elif compressor_number == 10:
        block_sizes = [x for x in range(500, 15500, 1000)]
        analyze_bwt_mtf_entropy(input_file, block_sizes)

    elif compressor_number == 11:
        prepared_input_file = prepare_input_file(input_file)
        with open(prepared_input_file, 'rb') as file:
            raw_data = file.read()
        try:
            text_data = raw_data.decode('utf-8')
            prepared_data = to_bytes(text_data)
        except:
            prepared_data = raw_data
        #buffer_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
        buffer_sizes = [2 ** i for i in range(0, 8)]
        ratios = test_lzss_compression(prepared_data, buffer_sizes)
        plt.plot(buffer_sizes, ratios, marker='o', color='orange')
        plt.xlabel("Размер буфера")
        plt.ylabel("Коэффициент сжатия")
        plt.title("Зависимость коэффициента сжатия LZSS от размера буфера")
        plt.grid(True)
        plt.show()

    elif compressor_number == 12:
        prepared_input_file = prepare_input_file(input_file)
        with open(prepared_input_file, 'rb') as file:
            raw_data = file.read()
        try:
            text_data = raw_data.decode('utf-8')
            prepared_data = to_bytes(text_data)
        except:
            prepared_data = raw_data
        #dict_sizes = [2 ** i for i in range(8, 15)]
        dict_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        ratios = test_lzw_compression(prepared_data, dict_sizes)
        plt.plot(dict_sizes, ratios, marker='o', color='orange')
        plt.xlabel("Размер словаря")
        plt.ylabel("Коэффициент сжатия")
        plt.title("Зависимость коэффициента сжатия LZW от размера словаря")
        plt.grid(True)
        plt.show()
    else:
        print("Неверный выбор компрессора...")

if __name__ == "__main__":
    run_selected_compressor()