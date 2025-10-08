from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from ..indicators.indicators import adx


class AdxStrategy(BaseStrategy):
    """Estrategia basada en el Índice de Movimiento Direccional Promedio (ADX).
    
    Parámetros:
        period: Período para el cálculo del ADX (por defecto 14)
        adx_threshold: Umbral mínimo de ADX para considerar la señal (por defecto 20)
        min_di_diff: Diferencia mínima entre +DI y -DI para considerar señal (por defecto 2.0)
        adx_slope_lookback: Período para calcular la pendiente del ADX (por defecto 3)
        require_cross: Si True, requiere cruce de +DI y -DI; si False, solo prioridad de +DI sobre -DI (por defecto True)
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.period = int(params.get('period', 14))
        self.adx_threshold = float(params.get('adx_threshold', 20))
        self.min_di_diff = float(params.get('min_di_diff', 2.0))
        self.adx_slope_lookback = int(params.get('adx_slope_lookback', 3))
        self.require_cross = bool(params.get('require_cross', True))

    def _calculate_slope(self, series: pd.Series, lookback: int) -> pd.Series:
        """Calcula la pendiente de una serie sobre un período de lookback"""
        return series.diff(lookback) / lookback

    def generate_signals(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        # Verificar que hay suficientes datos
        min_required = max(self.period * 2, 30)  # Mínimo 30 velas o 2*periodo
        if len(ohlcv) < min_required:
            return pd.DataFrame(index=ohlcv.index, columns=['signal']).fillna(0)
            
        df = ohlcv.copy()
        
        # 1. Calcular indicadores
        adx_data = adx(df, self.period)
        df = df.join(adx_data)
        
        # 2. Limpiar y preparar datos
        for col in ['ADX', '+DI', '-DI']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill()
        
        # 3. Calcular condiciones de tendencia
        df['adx_rising'] = df['ADX'] > df['ADX'].shift(1)
        df['adx_slope'] = self._calculate_slope(df['ADX'], self.adx_slope_lookback)
        
        # 4. Condiciones base
        di_cross_up = (df['+DI'] > df['-DI']) & (df['+DI'].shift(1) <= df['-DI'].shift(1))
        di_cross_down = (df['-DI'] > df['+DI']) & (df['-DI'].shift(1) <= df['+DI'].shift(1))
        di_above = df['+DI'] > (df['-DI'] + self.min_di_diff)
        di_below = df['-DI'] > (df['+DI'] + self.min_di_diff)
        adx_strong = df['ADX'] > self.adx_threshold
        adx_rising = df['adx_slope'] > 0  # ADX en aumento

        # 5. Generar señales (cruce opcional)
        if self.require_cross:
            # df['buy_signal'] = di_cross_up & di_above & adx_strong & adx_rising
            df['buy_signal'] = di_cross_up
            # df['sell_signal'] = di_cross_down & di_below & adx_strong & adx_rising
            df['sell_signal'] = di_cross_down
        else:
            df['buy_signal'] = (df['+DI'] > df['-DI']) & di_above & adx_strong & adx_rising
            df['sell_signal'] = (df['-DI'] > df['+DI']) & di_below & adx_strong & adx_rising
        
        # 8. Señal final (1 = compra, -1 = venta, 0 = sin señal)
        df['signal'] = 0
        df.loc[df['buy_signal'], 'signal'] = 1
        df.loc[df['sell_signal'], 'signal'] = -1
        
        # 9. Asegurarse de no tener señales en los primeros períodos
        df.loc[df.index[:min_required], 'signal'] = 0
        
        # 10. Asegurar que no haya señales en filas con valores NaN y tipo entero
        df['signal'] = df['signal'].fillna(0).astype(int)


        return df[['signal']]

