Projektübersicht zu OR-Warehouse Reinforcement Learning

Struktur:
 - src/
beinhaltet den allgemeinen Code für die Agenten, die Netzwerke und das Experience Memory

 - src/warehouse
beinhaltet den Code für die verschiedenen Experimente mit der Warehouse Implementation. In environment befindet sich
die Implementation der Warenhausumgebung. In evaluation sind Skripte zur Berechnung und zum Plotten verschiedener
Metriken und Graphen.

 - src/taxi
enthält den Trainingscode für die Taxi Umgebung.

Die Dateien mit "_play" starten einen Agenten auf Basis eines Checkpoints, um die Leistung
mitverfolgen und einschätzen zu können.

Es finden sich hier und da Elemente, die es nicht in die schriftliche Ausarbeitung geschafft haben:
 - LstmDQN
 - CombinedReplayMemory
Die jeweiligen Stellen sind markiert und der Ursprung der Idee angegeben.

Da die Parameter in jedem Skript vorkommen, fasse ich sie hier einmal zusammen:

hyperparameters = {
    # training
    'batch_size': 32,
    'learning_rate': 0.001,
    'scheduler_milestones': [30000, 60000],
    'scheduler_decay': 0.1, -> wird nach erreichen der milestones auf die lr multipliziert
    'optimizer': optimizer.Adam,
    'loss': F.smooth_l1_loss,
    'running_k': 4, -> Anzahl der Beobachtungen, die der RecurrentQAgent sammelt -> betrifft nur LstmDQN / RecurrentQAgent
    'bidirectional': False, -> bidirektionale LSTMs -> betrifft nur LstmDQN / RecurrentQAgent
    'combined_memory': False, -> aktiviert andere ReplayMemory Idee -> in memory.py vermerkt mit Quelle
    # reinforcement & environment
    'eps_policy': ExponentialEpsilonGreedyPolicy(eps_max=1.0, eps_min=0.02, decay=2000),
    'gamma': 0.9,
    'target_update': 10, -> Anzahl der Episode bis das TargetNetwork aktualisiert wird
    'num_episodes': 50001, -> Anzahl der Episoden im Training ( + 1, um sicherzustellen, dass wirklich 50.000 erreicht werden. vermutlich nicht mehr notwendig )
    'memory_capacity': 50000, -> Kapazität des Erfahrungsspeichers
    'warmup_episodes': 100, -> Anzahl der warm-up Episoden
    'save_freq': 25000, -> Anzahl der Episoden nach denen ein checkpoint erstellt wird ( sollte so gewählt werden, dass es num_episodes teilt )
    'max_steps_per_episode': 100, -> Schrittlimit pro Episode im Training
    # Größe des Warenhauses
    'num_aisles': 4, -> zählt nur die ailses in der mitte ohne den Rand
    'rack_height': 8,
    'min_packets': 4,
    'max_packets': 8, -> kann gleich min_packets gesetzt werden, um Paketanzahl konstant zu halten
    # pytorch
    'np_seed': seed,
    'device': 'cpu', -> kann auf 'cuda' gesetzt werden, falls vorhanden. Aufgrund der sequentiellen Natur des RL nur bei sehr komplexen Modellen wirklich sinnvoll
    'save_model': True,
    'dtype': torch.float32,
    'plot_progress': False, -> aktualisiert plots während des Trainings
    'ylim': (-200, 200), -> Größe des Plots
    'tag': f'warehouse_v7_thesis_big_vp_{seed}' -> legt eine Bezeichnung für die checkpoints fest
}