# GPU Arcade

FC 卡带风格的终端游戏合集，所有游戏逻辑在 GPU 上通过 OpenCL 并行执行。

![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 游戏列表

| # | 游戏 | GPU 负责 |
|---|------|---------|
| 1 | **Snake** | 蛇身老化、碰撞检测、食物生成 |
| 2 | **Tetris** | 消行检测（并行扫描所有行） |
| 3 | **Game of Life** | 细胞状态更新（每个格子一个 GPU 线程） |
| 4 | **Pong** | 球的物理运动、碰撞检测 |
| 5 | **Breakout** | 球的运动、砖块碰撞检测 |

## 截图

```
  ____  _   _  _____       _                _
 / ___|| | | ||  ___|__ _| | ___ __ _ _ __ | |
| |  _ | |_| || |_ / __| |/ __/ _` | '__|| |
| |_| ||  _  ||  _| (__| | (_| (_| | |   | |
 \____|_|_|_||_|  \___|_|_\___\__,_|_|   |_|

  GPU: Clover | AMD Radeon HD 8210 | 2 CU @ 300 MHz

  +------------------------------------------------+
  | > 1. Snake          Classic snake game          |
  |   2. Tetris         Stack falling blocks        |
  |   3. Game of Life   Cellular automaton          |
  |   4. Pong           Classic paddle ball         |
  |   5. Breakout       Break bricks with ball      |
  +------------------------------------------------+

  Up/Down=Select  Enter=Play  Q=Quit
```

## 下载

从 [Releases](https://github.com/Jasonlecson/gpu-arcade/releases) 下载。

## 从源码编译

### Linux

```bash
sudo apt install build-essential libncurses-dev ocl-icd-opencl-dev opencl-headers
make
./gpu_arcade
```

权限问题：

```bash
sg render -c "./gpu_arcade"
```

### macOS

```bash
xcode-select --install
make
./gpu_arcade
```

### Windows (MSYS2/MinGW64)

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-icd mingw-w64-x86_64-opencl-headers
make
./gpu_arcade.exe
```

## 操作

| 按键 | 功能 |
|------|------|
| 上/下 | 选择游戏 |
| 回车 | 进入游戏 |
| Q / ESC | 返回菜单 / 退出 |

各游戏内部操作见游戏内提示。

## 架构

```
gpu-arcade/
├── main.c                 # 主菜单（FC 风格选择界面）
├── src/
│   ├── common.h           # 共享框架（OpenCL、终端抽象、输入处理）
│   ├── game_snake.c       # Snake - 蛇身/碰撞/食物全在 GPU
│   ├── game_tetris.c      # Tetris - GPU 并行消行检测
│   ├── game_life.c        # Game of Life - 每格子一个 GPU 线程
│   ├── game_pong.c        # Pong - 球物理在 GPU
│   └── game_breakout.c    # Breakout - 碰撞在 GPU
├── Makefile               # 自动检测平台编译
├── .github/workflows/     # GitHub Actions 三平台自动构建
├── LICENSE
└── README.md
```

### 添加新游戏

1. 创建 `src/game_xxx.c`，实现 `int game_xxx(gpu_ctx_t *gpu)`
2. 在 `main.c` 的 `games[]` 数组中注册
3. Done

## 许可证

[MIT](LICENSE)
