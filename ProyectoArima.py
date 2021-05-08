import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot




class pronostico():
    
     def __init__(self, lista):
        self.__lista=lista


     def cargar_archivo(self):
        valores_acciones=pd.read_csv(self.__lista,index_col='Date',parse_dates=True)
        return valores_acciones



     def borrar_vacias(self):
        valores_acciones=self.cargar_archivo()
        return valores_acciones.dropna()



     def imprimir_dimension(self):
        valores_acciones=self.borrar_vacias()
        print('Dimension',valores_acciones.shape)
        valores_acciones.head()
        print(valores_acciones)
        valores_acciones['Close'].plot(figsize=(12,5))
        plt.show()



     def calculo_aic(self):
        valores_acciones=self.borrar_vacias()
        test = adfuller(valores_acciones['Close'], autolag = 'AIC')
        print("1. ADF : ",test[0])
        print("2. P-Value : ", test[1])
        print("3. Num Of Lags : ", test[2])
        print("4. Num Of Observations Used For ADF Regression:",      test[3])
        print("5. Critical Values :")
        for key, val in test[4].items():
            print("\t",key, ": ", val)



     def calculo_modelo(self):
        valores_acciones=self.borrar_vacias()
        modelo = auto_arima(valores_acciones['Close'], trace=True, suppress_warnings=True)
        print(valores_acciones.shape)



     def tomar_datos(self):
        valores_acciones=self.borrar_vacias()
        inicio=valores_acciones.iloc[:-80]
        final=valores_acciones.iloc[-80:]
        print(inicio.shape,final.shape)
        return inicio, final



     def metodo_arima(self):
        inicio, final= self.tomar_datos()
        model=ARIMA(inicio['Close'],order=(1,1,0))
        model=model.fit()
        model.summary()
        return model



     def calculo_residuos(self):
        model=self.metodo_arima()
        residuos = pd.DataFrame(model.resid)
        residuos.plot()
        pyplot.show()
        return residuos



     def calculo_densidad(self):
        residuos=self.calculo_residuos()
        residuos.plot(kind='kde')
        pyplot.show()
        print(residuos.describe())



     def imprimir_prediccion(self):
        model=self.metodo_arima()
        inicio, final= self.tomar_datos()
        prediccion=model.predict(typ='levels').rename('Predicciones ARIMA')
        prediccion.plot(legend=True)
        inicio['Close'].plot(legend=True)
        plt.show()
        return prediccion




forecasting= pronostico('./AMD.csv')

forecasting.imprimir_dimension()

forecasting.calculo_aic()

forecasting.calculo_modelo()

forecasting.tomar_datos()

forecasting.metodo_arima()

forecasting.imprimir_prediccion()

forecasting.calculo_residuos()

forecasting.calculo_densidad()


