#!/bin/bash

echo "💰 PROFIT ANALYSIS TOOL"
echo "======================="

LOG_FILES=$(ls trading_log_*.log autonomous_trader.log 2>/dev/null)

if [ -z "$LOG_FILES" ]; then
    echo "❌ No log files found. Bot hasn't been run yet."
    exit 1
fi

echo "📊 Analyzing trading performance..."
echo ""

# Extract trade information from logs
echo "🎯 TRADE SUMMARY:"
echo "=================="

EXECUTED_TRADES=$(grep "EXECUTED:" $LOG_FILES | wc -l)
CLOSED_TRADES=$(grep "CLOSED:" $LOG_FILES | wc -l)

echo "Total Executed Trades: $EXECUTED_TRADES"
echo "Total Closed Trades: $CLOSED_TRADES"

if [ $CLOSED_TRADES -gt 0 ]; then
    echo ""
    echo "💸 P&L ANALYSIS:"
    echo "==============="
    
    # Extract P&L values
    grep "CLOSED:" $LOG_FILES | grep -o "P&L: \$[0-9.-]*" | sed 's/P&L: \$//g' > /tmp/pnl_values.txt
    
    if [ -s /tmp/pnl_values.txt ]; then
        # Calculate total P&L
        TOTAL_PNL=$(awk '{sum+=$1} END {printf "%.4f", sum}' /tmp/pnl_values.txt)
        echo "Total P&L: \$$TOTAL_PNL"
        
        # Calculate win rate
        WINNING_TRADES=$(awk '$1 > 0' /tmp/pnl_values.txt | wc -l)
        WIN_RATE=$(echo "scale=2; $WINNING_TRADES * 100 / $CLOSED_TRADES" | bc 2>/dev/null || echo "N/A")
        echo "Win Rate: ${WIN_RATE}%"
        
        # Best and worst trades
        BEST_TRADE=$(sort -n /tmp/pnl_values.txt | tail -n 1)
        WORST_TRADE=$(sort -n /tmp/pnl_values.txt | head -n 1)
        echo "Best Trade: \$$BEST_TRADE"
        echo "Worst Trade: \$$WORST_TRADE"
        
        rm /tmp/pnl_values.txt
    fi
    
    echo ""
    echo "📈 RECENT ACTIVITY:"
    echo "=================="
    echo "Last 5 trades:"
    grep "CLOSED:" $LOG_FILES | tail -n 5
    
fi

echo ""
echo "🔍 CURRENT STATUS:"
echo "=================="
if pgrep -f "crypto_trading" > /dev/null; then
    echo "✅ Bot is currently RUNNING"
    echo "💰 Check balance in the live display"
else
    echo "❌ Bot is NOT running"
    echo "🚀 Start with: ./start_native_bot.sh"
fi

echo ""
echo "📊 For live monitoring: ./monitor_bot.sh"
