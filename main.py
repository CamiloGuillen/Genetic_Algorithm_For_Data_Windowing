import numpy as np
import matplotlib.pyplot as plt

from ga_class import GeneticAlgorithm

x = np.linspace(-np.pi, np.pi, 151)
signal_a = np.sin(x)
signal_b = np.cos(x)

ga = GeneticAlgorithm(data_signal_1=signal_a, data_signal_2=signal_b, max_pop=1000)
opt_window, opt_loss, best_windows_hist, avg_fitness_hist = ga.run(eps=0.01,
                                                                   stop_criteria_samples=5,
                                                                   max_iteration=100,
                                                                   mutation_rate=0.2,
                                                                   mutation_tolerance=10,
                                                                   mutation_range=range(-5, 5))

plt.figure()
for i, point in enumerate(best_windows_hist):
    plt.plot(range(len(x)), signal_a)
    plt.plot(range(len(x)), signal_b)
    plt.grid()
    plt.axvline(x=point[0], c='r', ls='--')
    plt.axvline(x=point[1], c='g', ls='--')
    plt.draw()
    plt.pause(0.01)
    plt.clf()

plt.plot(range(len(x)), signal_a)
plt.plot(range(len(x)), signal_b)
plt.grid()
plt.axvline(x=opt_window[0], c='r', ls='--')
plt.axvline(x=opt_window[1], c='g', ls='--')
plt.draw()

plt.figure()
plt.plot(range(len(avg_fitness_hist)), avg_fitness_hist)
plt.xlabel("Generation")
plt.ylabel("Avg Fitness Score")
plt.grid()
plt.show()
