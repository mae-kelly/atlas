#!/bin/bash

echo "🔧 CREATING MAC SYSTEM SERVICE"
echo "============================="

# Get the current directory
CURRENT_DIR=$(pwd)

# Create launchd plist
cat > ~/Library/LaunchAgents/com.autonomoustrader.plist << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.autonomoustrader</string>
    <key>ProgramArguments</key>
    <array>
        <string>${CURRENT_DIR}/start_native_bot.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${CURRENT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${CURRENT_DIR}/autonomous_trader.log</string>
    <key>StandardErrorPath</key>
    <string>${CURRENT_DIR}/autonomous_trader_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
PLIST_EOF

echo "✅ Service file created at ~/Library/LaunchAgents/com.autonomoustrader.plist"
echo ""
echo "🚀 TO START AS SYSTEM SERVICE:"
echo "   launchctl load ~/Library/LaunchAgents/com.autonomoustrader.plist"
echo ""
echo "🛑 TO STOP SERVICE:"
echo "   launchctl unload ~/Library/LaunchAgents/com.autonomoustrader.plist"
echo ""
echo "📊 TO VIEW LOGS:"
echo "   tail -f autonomous_trader.log"
echo ""
echo "⚡ This will run the trader automatically on Mac startup!"
