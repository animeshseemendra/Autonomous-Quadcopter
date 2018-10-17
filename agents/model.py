#Building deep model here
from keras import layers ,models , optimizers ,regularizers
from keras import backend as bk

class takeoff:
    def __init__(self,state_size,action_size,action_low,action_high):
            self.state_size=state_size
            self.action_size=action_size
            self.action_low=action_low
            self.action_high=action_high
            self.action_range=action_high-action_low
            
            self.build_model_actor()
            self.build_model_critic()
 
    def build_model_actor(self):
        states=layers.Input(shape=(self.state_size,),name='states')
        
        layer=layers.Dense(32,use_bias=False,kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(states)
        layer=layers.BatchNormalization()(layer)
        layer=layers.Activation('relu')(layer)
        layer=layers.Dropout(0.45)(layer)
        
        layer=layers.Dense(64,use_bias=False,kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(layer)
        layer=layers.BatchNormalization()(layer)
        layer=layers.Activation('relu')(layer)
        layer=layers.Dropout(0.45)(layer)
        
        layer=layers.Dense(128,use_bias=False,kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(layer)
        layer=layers.BatchNormalization()(layer)
        layer=layers.Activation('relu')(layer)
        layer=layers.Dropout(0.45)(layer)
        
        
   
        output=layers.Dense(self.action_size,activation='sigmoid',name='output')(layer)
        action=layers.Lambda(lambda x: self.action_low+self.action_range*x,name='action')(output)
        
        self.model = models.Model(inputs = states, outputs = action)
        
        return self.model,action
    def build_model_critic(self):
        
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, use_bias = False, kernel_regularizer = regularizers.l2(0.01), 
                                  activity_regularizer=regularizers.l1(0.01))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.45)(net_states)
        
        net_states = layers.Dense(units=64, use_bias = False, kernel_regularizer = regularizers.l2(0.01), 
                                  activity_regularizer = regularizers.l1(0.01))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.45)(net_states)
        
        net_states = layers.Dense(units=128, use_bias = False, kernel_regularizer = regularizers.l2(0.01), 
                                  activity_regularizer = regularizers.l1(0.01))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.45)(net_states)
        
        

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, use_bias = False, kernel_regularizer=regularizers.l2(0.01),
                                       activity_regularizer=regularizers.l1(0.01))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.45)(net_actions)
        
        net_actions = layers.Dense(units=64, use_bias = False, kernel_regularizer=regularizers.l2(0.01), 
                                   activity_regularizer=regularizers.l1(0.01))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.45)(net_actions)
                                          
        net_actions = layers.Dense(units=128, use_bias = False, kernel_regularizer=regularizers.l2(0.01), 
                                   activity_regularizer=regularizers.l1(0.01))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.45)(net_actions)
        
        

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        net=  layers.Dense(32,use_bias=False,kernel_regularizer=regularizers.l2(0.01),
                           activity_regularizer=regularizers.l1(0.01))(net)
        net=  layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=  layers.Dropout(0.45)(net)
      
        
        # Add final output layer to produce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        return self.model,actions,Q_values
    
    
        
        
        
        
