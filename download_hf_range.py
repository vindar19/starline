import os
import time
import requests

PROXY = "http://127.0.0.1:7897"

HF_URL = (
    "https://huggingface.co/"
    "kataragi/ControlNet-LineartXL/"
    "resolve/main/"
    "Katarag_lineartXL-fp16.safetensors"
)

OUTPUT = r"G:\paint assistant\ref_prj\starline\controlnet\lineart\Katarag_lineartXL-fp16.safetensors"

TOTAL_SIZE = 2502139104

# 每块 16 MB
CHUNK_SIZE = 16 * 1024 * 1024

# 单块失败后的重试次数
MAX_RETRY = 10

# 超时
TIMEOUT = 60


def download_chunk(start, end, output):
    expected = end - start + 1

    for retry in range(1, MAX_RETRY + 1):
        try:
            print(
                f"\n[{start:,}-{end:,}] "
                f"尝试 {retry}/{MAX_RETRY}"
            )

            proxies = {
                "http": PROXY,
                "https": PROXY,
            }

            headers = {
                "Range": f"bytes={start}-{end}",
                "User-Agent": "Mozilla/5.0",
            }

            with requests.get(
                HF_URL,
                headers=headers,
                proxies=proxies,
                stream=True,
                timeout=TIMEOUT,
                allow_redirects=True,
            ) as r:

                r.raise_for_status()

                status = r.status_code

                if status != 206:
                    raise RuntimeError(
                        f"服务器没有返回 206 Partial Content，"
                        f"实际状态码: {status}"
                    )

                content_range = r.headers.get("Content-Range", "")
                print(f"Content-Range: {content_range}")

                temp = output + ".part"

                received = 0

                with open(temp, "wb") as f:
                    for data in r.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not data:
                            continue

                        f.write(data)
                        received += len(data)

                if received != expected:
                    raise RuntimeError(
                        f"块大小不正确："
                        f"收到 {received:,} bytes，"
                        f"期望 {expected:,} bytes"
                    )

                os.replace(temp, output)

                print(
                    f"完成: {received / 1024 / 1024:.2f} MB"
                )

                return True

        except Exception as e:
            print(f"失败: {type(e).__name__}: {e}")

            temp = output + ".part"

            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except Exception:
                    pass

            if retry < MAX_RETRY:
                wait = min(retry * 3, 30)

                print(
                    f"{wait} 秒后重试..."
                )

                time.sleep(wait)

    return False


def main():

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    print("=" * 70)
    print("Hugging Face Range Downloader")
    print("=" * 70)

    print(f"目标文件:")
    print(OUTPUT)

    print(f"\n文件大小:")
    print(
        f"{TOTAL_SIZE:,} bytes "
        f"({TOTAL_SIZE / 1024 / 1024 / 1024:.2f} GB)"
    )

    print(f"\n分块大小:")
    print(
        f"{CHUNK_SIZE / 1024 / 1024:.0f} MB"
    )

    total_chunks = (
        TOTAL_SIZE + CHUNK_SIZE - 1
    ) // CHUNK_SIZE

    print(f"\n总块数: {total_chunks}")
    print("=" * 70)

    # 用一个临时目录保存已经完成的块
    chunk_dir = OUTPUT + ".chunks"

    os.makedirs(chunk_dir, exist_ok=True)

    for index, start in enumerate(
        range(0, TOTAL_SIZE, CHUNK_SIZE)
    ):

        end = min(
            start + CHUNK_SIZE - 1,
            TOTAL_SIZE - 1
        )

        chunk_file = os.path.join(
            chunk_dir,
            f"{index:05d}.part"
        )

        expected = end - start + 1

        # 如果块已经完整存在，就跳过
        if os.path.exists(chunk_file):

            size = os.path.getsize(chunk_file)

            if size == expected:
                print(
                    f"[{index + 1}/{total_chunks}] "
                    f"已存在，跳过"
                )
                continue

            else:
                os.remove(chunk_file)

        print(
            f"\n========== "
            f"块 {index + 1}/{total_chunks} "
            f"=========="
        )

        if not download_chunk(
            start,
            end,
            chunk_file
        ):
            print(
                "\n下载失败。"
            )
            print(
                "已经完成的块不会丢失。"
            )
            print(
                "重新运行本程序即可继续。"
            )
            return

        completed = min(
            end + 1,
            TOTAL_SIZE
        )

        percent = (
            completed / TOTAL_SIZE * 100
        )

        print(
            f"总体进度: "
            f"{percent:.2f}%"
        )

    print("\n")
    print("=" * 70)
    print("所有分块下载完成，开始合并")
    print("=" * 70)

    # 合并
    with open(OUTPUT, "wb") as out:

        for index, start in enumerate(
            range(0, TOTAL_SIZE, CHUNK_SIZE)
        ):

            chunk_file = os.path.join(
                chunk_dir,
                f"{index:05d}.part"
            )

            print(
                f"合并 {index + 1}/{total_chunks}"
            )

            with open(chunk_file, "rb") as f:
                while True:
                    data = f.read(8 * 1024 * 1024)

                    if not data:
                        break

                    out.write(data)

    # 最终大小验证
    final_size = os.path.getsize(OUTPUT)

    print("\n")
    print("=" * 70)
    print("最终检查")
    print("=" * 70)

    print(
        f"期望大小: {TOTAL_SIZE:,}"
    )

    print(
        f"实际大小: {final_size:,}"
    )

    if final_size != TOTAL_SIZE:

        print(
            "❌ 文件大小不正确！"
        )

        return

    print(
        "✅ 文件大小正确！"
    )

    print(
        "\n下载完成:"
    )

    print(OUTPUT)


if __name__ == "__main__":
    main()