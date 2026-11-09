#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_NAME="EasyColab"
EXEC_NAME="EasyColab"
BUNDLE_DIR="${PROJECT_ROOT}/${APP_NAME}.app"
CONTENTS_DIR="${BUNDLE_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

# 内置服务端与桥接资源，支持将 EasyColab.app 独立拖入 /Applications 开箱即用
for res_file in "app.py" "colab_bridge.py" "EasyColab_Server.ipynb" "requirements.txt"; do
    if [ -f "${PROJECT_ROOT}/${res_file}" ]; then
        cp -f "${PROJECT_ROOT}/${res_file}" "${RESOURCES_DIR}/${res_file}"
    fi
done
# colab_bridge.py 是薄入口：实现在 bridge/ 包里。app.py 是薄入口：服务端逻辑在 colab_server/ 包里；runtime/ 是随 app.py 上传的附加模块。桥接按 app.py 同目录查找它们
for res_dir in "bridge" "colab_server" "runtime"; do
    if [ -d "${PROJECT_ROOT}/${res_dir}" ]; then
        rm -rf "${RESOURCES_DIR}/${res_dir}"
        cp -R "${PROJECT_ROOT}/${res_dir}" "${RESOURCES_DIR}/${res_dir}"
        find "${RESOURCES_DIR}/${res_dir}" -name "__pycache__" -type d -prune -exec rm -rf {} +
    fi
done

# 1. 生成 Info.plist
cat > "${CONTENTS_DIR}/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${EXEC_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>app.easycolab.studio.macos</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundleVersion</key>
    <string>2</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsArbitraryLoads</key>
        <true/>
    </dict>
</dict>
</plist>
EOF

# 2. 生成 EasyColab 应用图标 (AppIcon.icns)
ICONSET_DIR="/tmp/EasyColabIcon.iconset"
rm -rf "${ICONSET_DIR}"
mkdir -p "${ICONSET_DIR}"

swift - << 'SWIFT_ICON'
import AppKit

func drawIcon(_ size: Int, _ path: String) {
    let s = CGFloat(size)
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)!
    let c = NSGraphicsContext.current!.cgContext
    c.scaleBy(x: s / 1024, y: s / 1024)
    c.setAllowsAntialiasing(true)
    let tile = CGRect(x: 92, y: 92, width: 840, height: 840)
    let shape = CGPath(roundedRect: tile, cornerWidth: 205, cornerHeight: 205, transform: nil)
    c.saveGState()
    c.setShadow(offset: CGSize(width: 0, height: -17), blur: 35, color: NSColor.black.withAlphaComponent(0.28).cgColor)
    c.addPath(shape); c.setFillColor(NSColor.black.cgColor); c.fillPath()
    c.restoreGState()
    c.saveGState(); c.addPath(shape); c.clip()
    let colors = [NSColor(calibratedRed: 0.81, green: 0.91, blue: 0.96, alpha: 1).cgColor,
                  NSColor(calibratedRed: 0.96, green: 0.99, blue: 1.00, alpha: 1).cgColor] as CFArray
    let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 1])!
    c.drawLinearGradient(gradient, start: CGPoint(x: 150, y: 100), end: CGPoint(x: 850, y: 940), options: [])
    c.restoreGState()
    // Center the cloud and its nodes together within the tile.
    c.saveGState()
    c.translateBy(x: -7, y: -48)
    // A single, generous cloud silhouette stays legible even at Dock size.
    let cloud = CGMutablePath()
    cloud.move(to: CGPoint(x: 322, y: 411))
    cloud.addCurve(to: CGPoint(x: 320, y: 624), control1: CGPoint(x: 247, y: 411), control2: CGPoint(x: 235, y: 600))
    cloud.addCurve(to: CGPoint(x: 617, y: 690), control1: CGPoint(x: 358, y: 773), control2: CGPoint(x: 558, y: 807))
    cloud.addCurve(to: CGPoint(x: 711, y: 604), control1: CGPoint(x: 670, y: 689), control2: CGPoint(x: 707, y: 650))
    cloud.addCurve(to: CGPoint(x: 710, y: 411), control1: CGPoint(x: 813, y: 598), control2: CGPoint(x: 828, y: 411))
    cloud.closeSubpath()
    c.saveGState()
    c.setShadow(offset: CGSize(width: 0, height: -12), blur: 22, color: NSColor(calibratedRed: 0.24, green: 0.48, blue: 0.64, alpha: 0.16).cgColor)
    c.addPath(cloud); c.setFillColor(NSColor.white.cgColor); c.fillPath()
    c.restoreGState()
    // Linked warm nodes represent the connection between a Mac and cloud compute.
    c.setStrokeColor(NSColor(calibratedRed: 0.95, green: 0.40, blue: 0.17, alpha: 1).cgColor)
    c.setLineWidth(34); c.setLineCap(.round); c.setLineJoin(.round)
    c.move(to: CGPoint(x: 405, y: 506)); c.addLine(to: CGPoint(x: 511, y: 579)); c.addLine(to: CGPoint(x: 620, y: 506)); c.strokePath()
    for point in [CGPoint(x: 405, y: 506), CGPoint(x: 511, y: 579), CGPoint(x: 620, y: 506)] {
        c.setFillColor(NSColor(calibratedRed: 1, green: 0.57, blue: 0.25, alpha: 1).cgColor)
        c.fillEllipse(in: CGRect(x: point.x - 32, y: point.y - 32, width: 64, height: 64))
    }
    c.restoreGState()
    NSGraphicsContext.restoreGraphicsState()
    try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
}
for size in [16, 32, 128, 256, 512] {
    drawIcon(size, "/tmp/EasyColabIcon.iconset/icon_\(size)x\(size).png")
    drawIcon(size * 2, "/tmp/EasyColabIcon.iconset/icon_\(size)x\(size)@2x.png")
}
SWIFT_ICON

iconutil -c icns "${ICONSET_DIR}" -o "${RESOURCES_DIR}/AppIcon.icns"
rm -rf "${ICONSET_DIR}"

# 3. 编译原生 SwiftUI macOS 通用可执行程序 (兼容 macOS 13.0+ Ventura/Sonoma/Sequoia/Tahoe/27 及 Apple Silicon + Intel)
echo "Compiling SwiftUI native macOS Universal app (${APP_NAME}, target macOS 13.0+)..."
swiftc -O \
    -target arm64-apple-macos13.0 \
    -parse-as-library \
    -framework SwiftUI \
    -framework AppKit \
    -framework AVKit \
    -framework UniformTypeIdentifiers \
    "${SCRIPT_DIR}/EasyColabStudio.swift" \
    -o "${MACOS_DIR}/${EXEC_NAME}_arm64"

if swiftc -O \
    -target x86_64-apple-macos13.0 \
    -parse-as-library \
    -framework SwiftUI \
    -framework AppKit \
    -framework AVKit \
    -framework UniformTypeIdentifiers \
    "${SCRIPT_DIR}/EasyColabStudio.swift" \
    -o "${MACOS_DIR}/${EXEC_NAME}_x86_64" 2>/dev/null; then
    lipo -create -output "${MACOS_DIR}/${EXEC_NAME}" \
        "${MACOS_DIR}/${EXEC_NAME}_arm64" \
        "${MACOS_DIR}/${EXEC_NAME}_x86_64"
    rm -f "${MACOS_DIR}/${EXEC_NAME}_arm64" "${MACOS_DIR}/${EXEC_NAME}_x86_64"
    echo "✅ Built Universal Binary (arm64 + x86_64, minos 13.0)"
else
    mv -f "${MACOS_DIR}/${EXEC_NAME}_arm64" "${MACOS_DIR}/${EXEC_NAME}"
    echo "✅ Built arm64 Binary (minos 13.0)"
fi

# 4. 本地签名并刷新 macOS LaunchServices 图标缓存
codesign --force --deep --sign - "${BUNDLE_DIR}"
touch "${BUNDLE_DIR}"

echo "✅ Successfully built: ${BUNDLE_DIR}"
