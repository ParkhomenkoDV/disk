import numpy as np
import pandas as pd
from numpy import array, full, zeros, nan, isnan, inf, pi, sqrt, arange, linspace
from scipy import interpolate, integrate
import matplotlib as mpl
import matplotlib.pyplot as plt

from substance import Material

# Список использованной литературы
REFERENCES = {
    1: '''Елисеев Ю.С., Крымов В.В., Манушин Э.А. и др.
Конструирование и расчет на прочность турбомашин ГТ и КУ:
Учебник для студентов вузов / Под общей ред. М.И. Осипова. – М.: МГТУ, 2009''',
    2: '''Иноземцев А.А. Динамика и прочность авиационных двигателей и энергетических установок:
учебник / А.А. Иноземцев, М.А. Нихамкин, В.Л. Сандрацкий. – М.: Машиностроение, 2008. – Т.2. – 368 с.; Т.4. – 192 с.''',
    3: '''Костюк А.Г. Динамика и прочность турбомашин:
Учебник для вузов. – 2-е изд., перераб. и доп. – М.: Издательство МЭИ, 2000. – 480 с.''',
    4: '''Щегляев А.В. Паровые турбины. – М.: Энергоатомиздат, 1993. – В двух книгах.''',
    5: '''Малинин Н.Н. Прочность турбомашин. – М.: Машиностроние, 1962. – 291 с.''',
    6: '''Колебания / В.С. Чигрин. - Пособие по лабораторному практикуму - Рыбинск: РГАТА, 2005. -20 с.''',
    7: '''Расчет дисков на прочность МАМИ''',
}


class Disk:
    __slots__ = ('__material', '__radius', '__thickness',
                 '__nholes', '__rholes', '__dholes', '__circumferential_distance')

    @classmethod
    def help(cls):
        print('Подбор толщин и радиусов для напряжения')

    def __init__(self, material: Material,
                 radius: tuple | list | np.ndarray, thickness: tuple | list | np.ndarray,
                 nholes: tuple | list | np.ndarray = tuple(),
                 rholes: tuple | list | np.ndarray = tuple(),
                 dholes: tuple | list | np.ndarray = tuple()) -> None:
        # проверка на тип данных material
        assert isinstance(material, Material)
        # проверка на тип данных radius и thickness
        assert isinstance(radius, (tuple, list, np.ndarray))
        assert isinstance(thickness, (tuple, list, np.ndarray))
        # проверка на тип данных nholes, rholes, dholes
        assert isinstance(nholes, (tuple, list, np.ndarray))
        assert isinstance(rholes, (tuple, list, np.ndarray))
        assert isinstance(dholes, (tuple, list, np.ndarray))
        # проверка на равенство массивов radius, thickness и nholes, rholes, dholes
        assert len(radius) == len(thickness)
        assert len(nholes) == len(rholes) == len(dholes)
        # проверка на тип данных внутри массивов radius, thickness
        assert all(isinstance(el, (int, float, np.number)) for el in radius)
        assert all(isinstance(el, (int, float, np.number)) for el in thickness)
        # проверка на тип данных внутри массивов nholes, rholes, dholes
        assert all(isinstance(el, (int, float, np.number)) for el in nholes)
        assert all(isinstance(el, (int, float, np.number)) for el in rholes)
        assert all(isinstance(el, (int, float, np.number)) for el in dholes)
        # проверка на адекватность данных radius, thickness
        assert all(map(lambda i: i >= 0, radius))
        assert all(map(lambda i: i >= 0, thickness))
        # проверка на адекватность данных nholes, rholes, dholes
        assert all(map(lambda i: i >= 0, nholes))
        assert all(map(lambda i: i >= 0, rholes))
        assert all(map(lambda i: i >= 0, dholes))

        # сортировка пузырьком радиусов по возрастанию вместе с соответствующими толщинами
        swapped = False
        for i in range(len(radius) - 1, 0, -1):
            for j in range(i):
                if radius[j] > radius[j + 1]:
                    radius[j], radius[j + 1] = radius[j + 1], radius[j]
                    thickness[j], thickness[j + 1] = thickness[j + 1], thickness[j]
                    swapped = True
            if swapped:
                swapped = False
            else:
                break

        # проверка на отсортированность radius по возрастанию
        assert all(radius[i] < radius[i + 1] for i in range(len(radius) - 1))
        # проверка на адекватность rholes по отношению к radius
        assert all(map(lambda i: radius[0] <= i <= radius[-1], rholes))
        # расчет и проверка расстояния между краями отверстий по окружности
        self.__circumferential_distance = array([2 * pi * rholes[i] / n - dholes[i] for i, n in enumerate(nholes)])
        assert all(map(lambda i: i > 0, self.__circumferential_distance))

        self.__material = material
        self.__radius, self.__thickness = array(radius), array(thickness)
        self.__nholes, self.__rholes, self.__dholes = array(nholes), array(rholes), array(dholes)

    def __setattr__(self, key, value):
        """Запрет на изменение атрибутов"""
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            raise Exception('Changing attributes is prohibited!')

    @property
    def material(self) -> Material:
        return self.__material

    @property
    def radius(self):
        return self.__radius

    @property
    def thickness(self):
        return self.__thickness

    @property
    def nholes(self):
        return self.__nholes

    @property
    def rholes(self):
        return self.__rholes

    @property
    def dholes(self):
        return self.__dholes

    @staticmethod
    def slicing(point0: tuple, point1: tuple, discreteness: int | np.integer) -> tuple[float, float]:
        """Дробление сечений ndis раз"""
        assert isinstance(point0, (tuple, list, np.ndarray)) and isinstance(point1, (tuple, list, np.ndarray))
        assert len(point0) == 2 and len(point1) == 2
        assert isinstance(discreteness, (int, np.integer)) and 1 <= discreteness
        k = (point1[1] - point0[1]) / (point1[0] - point0[0]) if (point1[0] - point0[0]) != 0 else inf
        b = point0[1] - k * point0[0]
        delta = (point1[0] - point0[0]) / discreteness
        x = point0[0] + delta * np.arange(discreteness)
        y = k * x + b
        return x, y

    @staticmethod
    def equivalent_energy_tension(sigma_t: float | int | np.number, sigma_r: float | int | np.number) -> float:
        """Эквивалентное напряжение по энергетической теории"""
        sigma1, sigma3 = max(sigma_t, sigma_r), min(sigma_t, sigma_r)
        return sqrt(sigma1 ** 2 - sigma1 * sigma3 + sigma3 ** 2)

    @staticmethod
    def calculation2(rotation_frequency: int | float | np.number, pressure: tuple | list | np.ndarray,
                     radius: np.ndarray, thickness: np.ndarray, tetta: np.ndarray,
                     av_density: np.ndarray, av_E: np.ndarray, av_mu: np.ndarray) -> dict[str: np.ndarray]:
        """Метод двух расчетов"""
        # проверка НУ
        assert isinstance(rotation_frequency, (float, int, np.number))
        assert isinstance(pressure, (tuple, list, np.ndarray)) and len(pressure) == 2
        assert all(isinstance(el, (int, float, np.number)) for el in pressure)
        # проверка на тип данных сечений
        assert isinstance(radius, (tuple, list, np.ndarray))
        assert isinstance(thickness, (tuple, list, np.ndarray))
        assert isinstance(tetta, (tuple, list, np.ndarray))
        # проверка на длину массивов сечений
        assert len(radius) == len(thickness) == len(tetta)
        # проверка на тип данных внутри массивов сечений
        assert all(isinstance(el, (int, float, np.number)) for el in radius)
        assert all(isinstance(el, (int, float, np.number)) for el in thickness)
        assert all(isinstance(el, (int, float, np.number)) for el in tetta)
        # проверка на тип данных средних характеристик участков
        assert isinstance(av_density, (tuple, list, np.ndarray))
        assert isinstance(av_E, (tuple, list, np.ndarray))
        assert isinstance(av_mu, (tuple, list, np.ndarray))
        # проверка на длину массивов средних характеристик участков
        assert len(av_density) == len(av_E) == len(av_mu)
        # проверка на тип данных внутри массивов средних характеристик участков
        assert all(isinstance(el, (int, float, np.number)) for el in av_density)
        assert all(isinstance(el, (int, float, np.number)) for el in av_E)
        assert all(isinstance(el, (int, float, np.number)) for el in av_mu)

        func_tetta = lambda r: (tetta[0] + (tetta[-1] - tetta[0]) * ((r - radius[0]) / (radius[-1] - radius[0])) ** 2)
        sigma_t = full((len(radius) - 1, 2), 400 * 10 ** 6)
        sigma_r = sigma_t.copy() if radius[0] == 0 else full((len(radius) - 1, 2), pressure[0])

        for i in range(len(radius) - 1):
            if i != 0:
                sigma_t[i][0] = (av_mu[i] * (thickness[i - 1] / thickness[i]) * sigma_r[i - 1][1] +
                                 (av_E[i] / av_E[i - 1]) * (sigma_t[i - 1][1] - av_mu[i - 1] * sigma_r[i - 1][1]))
                sigma_r[i][0] = (thickness[i - 1] / thickness[i]) * sigma_r[i - 1][1]
            c1 = (sigma_t[i][0] + sigma_r[i][0] +
                  (1 + av_mu[i]) / 2 * av_density[i] * (rotation_frequency * radius[i]) ** 2 +
                  av_E[i] * tetta[i]) / 2
            c2 = (sigma_t[i][0] - sigma_r[i][0] -
                  (1 - av_mu[i]) / 4 * av_density[i] * (rotation_frequency * radius[i]) ** 2 +
                  av_E[i] * tetta[i]) / 2 * radius[i] ** 2
            I = integrate.quad(lambda r: func_tetta(r) * r, radius[i], radius[i + 1])[0]
            sigma_t[i][1] = ((c1 + c2 / radius[i + 1] ** 2 -
                              (1 + 3 * av_mu[i]) / 8 * av_density[i] * (rotation_frequency * radius[i + 1]) ** 2 +
                              (av_E[i] / radius[i + 1] ** 2) * I) - av_E[i] * tetta[i + 1])
            sigma_r[i][1] = ((c1 - c2 / radius[i + 1] ** 2) -
                             (3 + 1 * av_mu[i]) / 8 * av_density[i] * (rotation_frequency * radius[i + 1]) ** 2 -
                             (av_E[i] / radius[i + 1] ** 2) * I)

        return {'tension_t': sigma_t, 'tension_r': sigma_r}

    def tensions(self, rotation_frequency: float | int | np.number, temperature0: int | float | np.number,
                 pressure: tuple | list | np.ndarray, temperature: tuple | list | np.ndarray,
                 discreteness: int = 10, show: bool = False) -> dict[str:  np.ndarray]:
        """Расчет напряжений в диске"""

        assert isinstance(temperature0, (float, int, np.number)) and temperature0 > 0
        assert isinstance(pressure, (tuple, list, np.ndarray))
        assert isinstance(temperature, (tuple, list, np.ndarray))
        assert all(isinstance(el, (int, float, np.number)) for el in pressure)
        assert all(isinstance(el, (int, float, np.number)) for el in temperature)
        assert len(pressure) == 2
        assert len(temperature) == 2 or len(temperature) == len(self.radius)
        assert isinstance(discreteness, (int, np.integer)) and discreteness >= 1
        assert isinstance(show, bool)

        tetta = (self.material.alpha((temperature[0] + temperature0) / 2) * (temperature[0] - temperature0),
                 self.material.alpha((temperature[-1] + temperature0) / 2) * (temperature[-1] - temperature0))
        # параболический закон распределения деформаций
        func_tetta = lambda r: \
            (tetta[0] + (tetta[-1] - tetta[0]) * ((r - self.radius[0]) / (self.radius[-1] - self.radius[0])) ** 2)

        if len(temperature) == len(self.radius):
            func_temperature = interpolate.interp1d(self.radius, temperature, kind=3, fill_value='extrapolate')
        else:
            func_temperature = lambda r: (temperature[0] + (temperature[-1] - temperature[0]) *
                                          ((r - self.radius[0]) / (self.radius[-1] - self.radius[0])) ** 2)

        radius, thickness = np.empty(0), np.empty(0)
        for i in range(len(self.radius) - 1):
            r, th = self.slicing((self.radius[i], self.thickness[i]), (self.radius[i + 1], self.thickness[i + 1]),
                                 discreteness)
            radius = np.concatenate((radius, r))
            thickness = np.concatenate((thickness, th))

        average_radius = array([(radius[i] + radius[i + 1]) / 2 for i in range(len(radius) - 1)])
        avarege_temperature = [func_temperature(av_r) for av_r in average_radius]
        avarege_density = array([self.material.density(av_t) for av_t in avarege_temperature])
        avarege_E = array([self.material.E(av_t) for av_t in avarege_temperature])
        avarege_mu = array([self.material.mu(av_t) for av_t in avarege_temperature])
        tetta = array([func_tetta(r) for r in radius])

        calc1 = self.calculation2(rotation_frequency, pressure,
                                  radius, thickness, tetta,
                                  avarege_density, avarege_E, avarege_mu)
        calc2 = self.calculation2(0, pressure,
                                  radius, thickness, zeros(len(radius)),
                                  avarege_density, avarege_E, avarege_mu)

        k = (pressure[-1] - calc1['tension_r'][-1][-1]) / calc2['tension_r'][-1][-1]  # коэффициент Мора

        sigma_t = calc1['tension_t'] + k * calc2['tension_t']
        sigma_r = calc1['tension_r'] + k * calc2['tension_r']

        sigma_t = array([sigma_t[0][0]] +
                        [(sigma_t[i][1] + sigma_t[i + 1][0]) / 2 for i in range(0, len(radius) - 2)] +
                        [sigma_t[-1][-1]])
        sigma_r = array([sigma_r[0][0]] +
                        [(sigma_r[i][1] + sigma_r[i + 1][0]) / 2 for i in range(0, len(radius) - 2)] +
                        [sigma_r[-1][-1]])

        sigma = array([self.equivalent_energy_tension(sigma_t[i], sigma_r[i]) for i in range(len(radius))])

        result = {'radius': radius, 'thickness': thickness,
                  'tension': sigma, 'tension_t': sigma_t, 'tension_r': sigma_r}

        if show: self.__show_tension(rotation_frequency, temperature0, result)

        return result

    @staticmethod
    def local_tension(n: int | np.integer, radius: int | float | np.number, diameter: int | float | np.number,
                      sigma_t: float | int | np.number, sigma_r: float | int | np.number) -> tuple[float, float]:
        """Местное напряжение от отверстия"""
        assert isinstance(n, (int, np.integer)) and 0 < n
        assert isinstance(radius, (float, int, np.number)) and 0 <= radius
        assert isinstance(diameter, (float, int, np.number)) and 0 < diameter
        assert isinstance(sigma_t, (float, int, np.number)) and isinstance(sigma_r, (float, int, np.number))

        b = 2 * pi * radius / n - diameter  # расчет расстояния между краями отверстий по окружности
        assert 0 < b, f'circumferential distance must be > 0, but equal {b}'
        if sigma_t != 0:  # ZeroDevisionError
            k = 3 - diameter / b - sigma_r / sigma_t
            sigma_t_hole = float(k * sigma_t)
            return sigma_t_hole * 1.1, sigma_t_hole * 1.15
        else:
            return nan, nan

    def __show_tension(self, rotation_frequency: float, temperature0: int | float, tensions: dict, **kwargs) -> None:
        """Визуализация напряжений"""
        radius, thickness = tensions.get('radius') * 1_000, tensions.get('thickness') * 1_000  # приведение к [мм]
        for key in tensions:
            if key.startswith('tension'):
                tensions[key] = tensions[key] / 10 ** 6  # приведение к [МПа]

        l, k = max(radius), 1.2
        ylim = 0 - l * (k - 1) / 2, l + l * (k - 1) / 2
        func = interpolate.interp1d(radius, thickness, kind=1)

        fg = plt.figure(figsize=kwargs.pop('figsize', (16, 8)))
        gs = fg.add_gridspec(1, 2)  # строки, столбцы

        fg.add_subplot(gs[0, 0])
        plt.title("Disk", fontsize=14, fontweight='bold')

        plt.plot(thickness / 2, radius, -thickness / 2, radius,
                 color='black', linestyle='solid', linewidth=3)
        plt.plot([-thickness[-1] / 2, thickness[-1] / 2], [radius[-1], radius[-1]],
                 color='black', linestyle='solid', linewidth=3)  # периферия
        if radius[0] > 0:
            plt.plot([-thickness[0] / 2, thickness[0] / 2], [radius[0]] * 2,
                     color='black', linestyle='solid', linewidth=3)  # втулка
        for i in range(len(radius) - 1):  # сечения
            av_th = (thickness[i] + thickness[i + 1]) / 2
            plt.gca().add_patch(mpl.patches.Rectangle((-av_th / 2, radius[i]),
                                                      av_th, radius[i + 1] - radius[i],
                                                      angle=0, rotation_point='xy', alpha=0.5, edgecolor='blue'))
        plt.plot([-thickness[0] / k, thickness[0] / k], [0] * 2,
                 color='orange', linestyle='dashdot', linewidth=1.5)  # ось вращения
        for r, d in zip(self.rholes, self.dholes):
            r, d = r * 1_000, d * 1000
            plt.plot((-func(r) / k, func(r) / k), [r] * 2, color='orange', linestyle='dashdot', linewidth=1.5)
            plt.plot((-func(r) / 2, func(r) / 2), [r + d / 2] * 2, (-func(r) / 2, func(r) / 2), [r - d / 2] * 2,
                     color='black', linestyle='dashed', linewidth=2)

        # НУ
        plt.arrow(x=0, y=radius[-1], dx=0, dy=l * (k - 1) / 4 * np.sign(tensions['tension_r'][-1]),
                  color='red', width=1)
        plt.annotate(f'Pressure [MPa]: {tensions["tension_r"][-1]:.1f}',
                     xy=(3, radius[-1] + l * (k - 1) / 4 * np.sign(tensions['tension_r'][-1])),
                     fontsize=12, va='center')
        if radius[0] != 0 and tensions["tension_r"][0] != 0:
            plt.arrow(x=0, y=radius[0], dx=0, dy=l * (k - 1) / 4 * np.sign(tensions['tension_r'][0]),
                      color='red', width=1)
            plt.annotate(f'Pressure [MPa]: {tensions["tension_r"][0]:.1f}',
                         xy=(3, radius[0] + l * (k - 1) / 4 * np.sign(tensions['tension_r'][0])),
                         fontsize=12, va='center')

        plt.grid(True)
        plt.axis('equal')
        plt.ylim(ylim)
        plt.xlabel("Thickness", fontsize=12)
        plt.ylabel("Radius", fontsize=12)
        plt.legend([f'rotation frequency [Hz]: {rotation_frequency:.1f}',
                    f'temperature [K]: {temperature0:.1f}'], fontsize=12)

        fg.add_subplot(gs[0, 1])
        plt.title('Tension', fontsize=14, fontweight='bold')

        plt.plot(tensions['tension'], radius, label='equivalent', color='black', linestyle='solid', linewidth=3)
        plt.plot(tensions['tension_t'], radius, label='tangential', color='green', linestyle='solid', linewidth=2)
        plt.plot(tensions['tension_r'], radius, label='radial', color='blue', linestyle='solid', linewidth=2)

        plt.legend(fontsize=12)
        plt.grid(True)
        plt.ylim(ylim)
        plt.xlabel("Tension", fontsize=12)
        plt.ylabel("Radius", fontsize=12)

        plt.tight_layout()
        plt.show()

    def show(self, **kwargs) -> None:
        """Визуализация диска"""
        radius, thickness = self.radius * 1_000, self.thickness * 1_000  # приведение к [мм]
        func = interpolate.interp1d(radius, thickness, kind='linear')
        k = 1.5

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
        plt.title(kwargs.pop('title', "Disk"), fontsize=14, fontweight='bold')

        plt.plot(thickness / 2, radius, -thickness / 2, radius,
                 color='black', linestyle='solid', linewidth=3)
        plt.plot([-thickness[-1] / 2, thickness[-1] / 2], [radius[-1], radius[-1]],
                 color='black', linestyle='solid', linewidth=3)  # периферия
        if radius[0] > 0:
            plt.plot([-thickness[0] / 2, thickness[0] / 2], [radius[0], radius[0]],
                     color='black', linestyle='solid', linewidth=3)  # втулка
        plt.plot([-thickness[0] / k, thickness[0] / k], [0, 0],
                 color='orange', linestyle='dashdot', linewidth=1.5)  # ось вращения
        for r, d in zip(self.rholes, self.dholes):
            r, d = r * 1_000, d * 1000
            plt.plot((-func(r) / k, func(r) / k), [r] * 2, color='orange', linestyle='dashdot', linewidth=1.5)
            plt.plot((-func(r) / 2, func(r) / 2), [r + d / 2] * 2, (-func(r) / 2, func(r) / 2), [r - d / 2] * 2,
                     color='black', linestyle='dashed', linewidth=2)

        plt.grid(True)
        plt.axis('equal')
        plt.xlabel(kwargs.get('xlabel', 'Thickness'), fontsize=12)
        plt.ylabel(kwargs.get('ylabel', 'Radius'), fontsize=12)
        plt.tight_layout()
        plt.show()

    def equal_strength(self, tension: int | float | np.number,
                       rotation_frequency: int | float | np.number, temperature: int | float | np.number,
                       radius: int | float | np.number = None, thickness: int | float | np.number = None,
                       discreteness: int = 100, show: bool = True, **kwargs):
        """
        Профилирование сплошного, равнопрочного, равномерно нагретого диска без центрального отверстия,
        нагруженного только силами инерции [7]
        """
        assert isinstance(tension, (int, float, np.number))
        assert isinstance(rotation_frequency, (int, float, np.number))
        assert isinstance(temperature, (int, float, np.number)) and 0 < temperature
        assert (isinstance(radius, (int, float, np.number)) and 0 <= radius) or radius is None
        assert (isinstance(thickness, (int, float, np.number)) and 0 < thickness) or thickness is None
        assert isinstance(discreteness, (int, np.integer))
        assert isinstance(show, bool)

        if radius is None: radius = self.radius[-1]  # если не указан конкретный радиус
        if thickness is None: thickness = self.thickness[-1]  # если не указана конкретная толщина

        equal_strength = lambda r: (
                thickness * np.exp(0.5 * self.material.density(temperature) * rotation_frequency ** 2 *
                                   (radius ** 2 - r ** 2) / tension))

        if show:
            r = linspace(0, self.radius[-1], discreteness)
            equal_strength_disk = Disk(material=self.material, radius=r, thickness=equal_strength(r))
            equal_strength_disk.show(title=kwargs.pop('title', 'Equal strength disk'))

        return equal_strength

    def frequency_safety_factor(self, rotation_frequency: int | float,
                                temperature: int | float,
                                pressure: tuple | list | np.ndarray) -> tuple[tuple[float, float], str]:
        """Запас по разрушающей частоте [2, c. 99]"""
        assert isinstance(rotation_frequency, (int, float))
        assert isinstance(temperature, (int, float)) and 0 < temperature
        assert isinstance(pressure, (tuple, list, np.ndarray))
        assert len(pressure) == 2

        func_temperature = lambda r: temperature
        func_thickness = interpolate.interp1d(self.radius, self.thickness, kind='linear')

        safety = integrate.quad(lambda r: self.material.sigma_s(func_temperature(r)) * func_thickness(r),
                                self.radius[0], self.radius[-1], points=self.radius)[0]

        for n, r, d in zip(self.nholes, self.rholes, self.dholes):
            safety += (self.material.sigma_s(func_temperature(r)) * r * func_thickness(r) *
                       (1 - (n * d) / (2 * pi * r)))
        safety /= (pressure[-1] * self.radius[-1] * self.thickness[-1] +
                   (rotation_frequency ** 2 *
                    integrate.quad(lambda r: self.material.density(func_temperature(r)) * func_thickness(r) * r ** 2,
                                   self.radius[0], self.radius[-1], points=self.radius)[0]))
        safety = float(sqrt(safety))
        return (safety * 0.9, safety * 0.95), ''  # действительный интервал запаса разрушающей частоты

    def natural_frequencies(self, radius: int | np.integer, N: int, S: int) -> tuple[float, str]:
        """Частота собственных колебаний [6]"""

        def alpha(radius: int, N: int, S: int) -> tuple[float, str]:
            """
            radius = тип крепления (0 = центральное, -1 = периферийное)
            N = число узловых диаметров
            S = число узловых окружностей
            """
            if radius == 0:  # тип крепления центральное
                table = [[3.75, 3.42, 5.39, 12.49],
                         [20.91, 27.56, 34.80, 53.30],
                         [60.68, nan, nan, nan]]
                return table[S][N] if 0 <= S <= 2 and 0 <= N <= 3 else nan
            elif radius == -1:  # тип крепления периферийное
                table = [[10.24, 21.25, 33.60, 51],
                         [39.80, 60.80, 84.60, 111],
                         [89.00, 120.0, 153.8, 190],
                         [158.3, 199.0, 243.0, nan]]
                return table[S][N] if 0 <= S <= 3 and 0 <= N <= 3 else nan
            else:
                raise Exception('radius in (0, -1)')  # тип крепления

        f = self.material.E(5) * np.mean(self.thickness) ** 2
        f /= 12 * (1 - self.material.mu(0) ** 2) * self.material.density(0)
        f = sqrt(f) * alpha(radius, N, S) / (2 * pi * self.radius[-1] ** 2)
        return float(f), '1/s'

    def campbell_diagram(self, radius: int | np.integer, N: int | np.integer, S: int | np.integer,
                         max_rotation_frequency: int | float | np.number,
                         multiplicity: tuple | list | np.ndarray = arange(1, 11, 1),
                         **kwargs) -> tuple[list[float], str]:
        """Диаграмма Кэмпбелла [6]"""
        assert isinstance(max_rotation_frequency, (int, float, np.number))
        assert isinstance(multiplicity, (list, tuple, np.ndarray))
        assert all(map(lambda i: isinstance(i, (int, np.integer)), multiplicity))
        assert all(map(lambda i: i >= 0, multiplicity))

        def B(radius: int, N: int, S: int):
            """
            radius = тип крепления (0 = центральное, -1 = периферийное)
            N = число узловых диаметров
            S = число узловых окружностей
            """
            if radius == 0:  # тип крепления центральное
                table = ((0, 1, 2.35, 4.0),
                         (3.3, 5.95, 8.95, 12.3))
                return table[S][N] if 0 <= S <= 1 and 0 <= N <= 3 else nan
            else:
                raise Exception('radius in (0, -1)')  # тип крепления

        rotation_frequency = np.arange(0, max_rotation_frequency + 1, 1) * (2 * pi)  # перевод из рад/c в 1/c=об/c=Гц
        # динамическая частота колебаний вращающегося диска
        f = sqrt(self.natural_frequencies(radius, N, S)[0] ** 2 + B(radius, N, S) * rotation_frequency ** 2)
        # волны бегущие по и против вращения диска
        f_plus, f_minus = f + N * rotation_frequency, f - N * rotation_frequency
        resonance = set()  # резонансные частоты [1/с]

        plt.figure(figsize=kwargs.pop('figsize', (8, 8)))
        plt.title(kwargs.pop('title', 'Campbell diagram'), fontsize=14, fontweight='bold')
        for k in multiplicity:
            plt.plot([0, rotation_frequency[-1]], [0, rotation_frequency[-1] * k],
                     color='orange', linestyle='solid', linewidth=1)
            plt.text(rotation_frequency[-1], rotation_frequency[-1] * k, f'k{k}',
                     fontsize=12, ha='left', va='center')
            if k ** 2 - B(radius, N, S) >= 0:
                x0 = f[0] / sqrt(k ** 2 - B(radius, N, S))  # f
                if not isnan(x0) and x0 <= rotation_frequency[-1]:
                    resonance.add(round(x0, 6))
                    plt.scatter(x0, k * x0, color='red')
            if k ** 2 - B(radius, N, S) + N ** 2 + 2 * k * N >= 0:
                x0 = f[0] / sqrt(k ** 2 - B(radius, N, S) + N ** 2 + 2 * k * N)  # f_minus
                if not isnan(x0) and x0 <= rotation_frequency[-1]:
                    resonance.add(round(x0, 6))
                    plt.scatter(x0, k * x0, color='red')
            if k ** 2 - B(radius, N, S) + N ** 2 - 2 * k * N >= 0:
                x0 = f[0] / sqrt(k ** 2 - B(radius, N, S) + N ** 2 - 2 * k * N)  # f_plus
                if not isnan(x0) and x0 <= rotation_frequency[-1]:
                    resonance.add(round(x0, 6))
                    plt.scatter(x0, k * x0, color='red')
        plt.plot(rotation_frequency, f, color='black', linestyle='solid', linewidth=2, label='f')
        plt.plot(rotation_frequency, f_plus, color='green', linestyle='solid', linewidth=1.5, label='f+')
        plt.plot(rotation_frequency, f_minus, color='blue', linestyle='solid', linewidth=1.5, label='f-')
        plt.xlabel(kwargs.pop('xlabel', 'Frequency [1/s]'), fontsize=12)
        plt.ylabel(kwargs.pop('ylabel', 'Frequency [1/s]'), fontsize=12)
        plt.grid(kwargs.pop('grid', True))
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        return sorted(list(map(float, resonance)), reverse=False), '1/s'


def test() -> None:
    """Тестирование"""
    disks, conditions = list(), list()

    if 1:
        material = Material('10Х11Н20ТЗР',
                            {
                                "density": 8400,
                                "alpha": 18 * 10 ** -6,
                                "E": interpolate.interp1d(arange(400, 800 + 1, 100),
                                                          array([1.74, 1.66, 1.57, 1.47, 1.32]) * 10 ** 11,
                                                          kind=3, bounds_error=False, fill_value='extrapolate'),
                                "mu": interpolate.interp1d(arange(400, 800 + 1, 100),
                                                           [0.384, 0.379, 0.371, 0.361, 0.347],
                                                           kind=3, bounds_error=False, fill_value='extrapolate'),
                                "sigma_s": 900 * 10 ** 6
                            })
        radius = array([20, 26, 30.62, 37.26, 56.94, 60.67, 72.95, 75.95, 102.41, 106.52, 109.82]) / 1000
        thickness = array([36, 36, 15.43, 11.27, 10, 12, 12, 8, 6, 11, 11]) / 1000
        nholes, rholes, dholes = [5], [66.8 / 1000], [6.2 / 1000]

        rotation_frequency = 2806.2
        temperature0 = 293.15
        pressure = (0, 120.6 * 10 ** 6)
        temperature = (350, 650)

        disks.append(Disk(material=material,
                          radius=radius, thickness=thickness,
                          nholes=nholes, rholes=rholes, dholes=dholes))
        conditions.append(dict(rotation_frequency=rotation_frequency, temperature0=temperature0,
                               pressure=pressure, temperature=temperature))

    if 1:
        material = Material('10Х11Н20ТЗР',
                            {
                                "density": 8000,
                                "alpha": 20 * 10 ** -6,
                                "E": 1.3 * 10 ** 11,
                                "mu": 0.33,
                                "sigma_s": 600 * 10 ** 6
                            })
        radius = array(
            [0, 272, 434, 506.5, 509, 584, 619, 621.5, 726, 737, 748, 763, 775, 860, 864, 868, 880, 884, 887,
             905]) / 1000
        thickness = array(
            [132, 132, 83, 88.5, 154, 154, 88.5, 83, 90, 107, 122, 96, 83, 89, 96, 127, 96, 89, 83, 83]) / 1000
        nholes, rholes, dholes = [10], [544 / 1000], [40 / 1000]

        rotation_frequency = 3000 * 2 * pi / 60
        temperature0 = 620
        pressure = (0, 150 * 10 ** 6)
        temperature = (620, 800)

        disks.append(Disk(material=material,
                          radius=radius, thickness=thickness,
                          nholes=nholes, rholes=rholes, dholes=dholes))
        conditions.append(dict(rotation_frequency=rotation_frequency, temperature0=temperature0,
                               pressure=pressure, temperature=temperature))

    if 1:
        material = Material('10Х11Н20ТЗР',
                            {
                                "density": 8200,
                                "alpha": 18 * 10 ** -6,
                                "E": 1.6 * 10 ** 11,
                                "mu": 0.33,
                                "sigma_s": 900 * 10 ** 6
                            })
        radius = array(
            [84.0, 95.1, 107.7, 120.2, 132.8, 145.4, 157.9, 170.6, 176.6, 189.2, 203.1, 218.9, 233.3, 245.9, 258.4,
             271.9, 277.8, 283.5, 289.3, 303]) / 1000
        thickness = array(
            [83, 83, 91, 69, 64, 58, 58, 58, 48, 43, 39, 35.8, 31, 27.8, 29.6, 35.6, 40, 50, 62, 62]) / 1000
        nholes, rholes, dholes = [], [], []

        rotation_frequency = 9150 * 2 * pi / 60
        temperature0 = 300
        pressure = (0, 110 * 10 ** 6)
        temperature = (800, 1050)

        disks.append(Disk(material=material,
                          radius=radius, thickness=thickness,
                          nholes=nholes, rholes=rholes, dholes=dholes))
        conditions.append(dict(rotation_frequency=rotation_frequency, temperature0=temperature0,
                               pressure=pressure, temperature=temperature))

    for disk, condition in zip(disks, conditions):
        print(pd.DataFrame({'radius': disk.radius, 'thickness': disk.thickness}))
        disk.show()

        tensions = disk.tensions(**condition, discreteness=10, show=True)
        f_sigma_t = interpolate.interp1d(tensions['radius'], tensions['tension_t'], kind=1)
        f_sigma_r = interpolate.interp1d(tensions['radius'], tensions['tension_r'], kind=1)
        for i in range(len(disk.nholes)):
            local_tension = disk.local_tension(disk.nholes[i],  disk.rholes[i], disk.dholes[i],
                                               float(f_sigma_t(disk.rholes[i])), float(f_sigma_r(disk.rholes[i])))
            print(f'holes: {i}, nholes []: {disk.nholes[i]}, rholes: {disk.rholes[i]}, dholes: {disk.dholes[i]}')
            print(f'tension_t in {local_tension}')

        equal_strength = disk.equal_strength(400 * 10 ** 6,
                                             condition["rotation_frequency"], max(condition['temperature']),
                                             discreteness=10, show=False)
        radius_equal_strength = linspace(0, disk.radius[-1], 10, endpoint=True)
        thickness_equal_strength = equal_strength(radius_equal_strength)
        disk_equal_strength = Disk(material=disk.material,
                                   radius=radius_equal_strength, thickness=thickness_equal_strength)
        disk_equal_strength.tensions(rotation_frequency=condition['rotation_frequency'],
                                     temperature0=condition['temperature0'],
                                     pressure=(0, 0), temperature=(700, 700),
                                     discreteness=10, show=True)
        disk_equal_strength.tensions(**condition, discreteness=10, show=True)

        print(f'frequency_safety_factor: '
              f'{disk.frequency_safety_factor(condition["rotation_frequency"], temperature=600, pressure=pressure)}')
        for name, fastening in zip(('РК', 'НА'), (0, -1)):
            print(f'{name}: {fastening = }')
            for d in range(0, 3 + 1, 1):  # число узловых диаметров
                for c in range(0, 3 + 1, 1):  # число узловых окружностей
                    print('\t' + f'{d = } {c = } {disk.natural_frequencies(fastening, d, c)}')
        resonance = disk.campbell_diagram(0, 1, 1, condition["rotation_frequency"] * 1.1,
                                          multiplicity=arange(1, 11, 1))
        print(resonance)


if __name__ == "__main__":
    import cProfile

    cProfile.run('test()', sort='cumtime')
