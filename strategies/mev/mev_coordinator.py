import asyncio
from .flashloan_arbitrage import FlashLoanArbitrage,SandwichAttack,CrossDEXArbitrage,LiquidationBot
from loguru import logger

class MEVCoordinator:
    def __init__(self,config:dict):
        self.flashloan_bot=FlashLoanArbitrage(config['web3_provider'],config['private_key'])
        self.sandwich_bot=SandwichAttack(config['web3_provider'])
        self.arbitrage_bot=CrossDEXArbitrage()
        self.liquidation_bot=LiquidationBot()
        self.running=False
    
    async def start_mev_operations(self):
        self.running=True
        logger.info("⚡ Starting MEV operations")
        await asyncio.gather(
            self._flashloan_loop(),
            self._sandwich_loop(),
            self._arbitrage_loop(),
            self._liquidation_loop()
        )
    
    async def _flashloan_loop(self):
        while self.running:
            try:
                opportunities=await self.flashloan_bot.scan_arbitrage_opportunities()
                for opp in opportunities:
                    await self.flashloan_bot.execute_flashloan_arbitrage(opp)
            except Exception as e:logger.error(f"❌ Flash loan loop error: {e}")
            await asyncio.sleep(1)
    
    async def _sandwich_loop(self):
        await self.sandwich_bot.monitor_mempool()
    
    async def _arbitrage_loop(self):
        while self.running:
            try:
                opportunities=await self.arbitrage_bot.scan_cross_dex_opportunities()
                for opp in opportunities:
                    logger.info(f"📊 Cross-DEX opportunity: {opp['token']} spread {opp['spread']:.2%}")
            except Exception as e:logger.error(f"❌ Arbitrage loop error: {e}")
            await asyncio.sleep(2)
    
    async def _liquidation_loop(self):
        await self.liquidation_bot.monitor_liquidations()
    
    def stop_mev_operations(self):
        self.running=False
        logger.info("⏹️ MEV operations stopped")
